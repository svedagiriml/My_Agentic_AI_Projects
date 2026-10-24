"""The seven subagents. Each is a ReAct agent with its own MCP tool allow-list,
role prompt, deterministic fallback planner and typed output contract.
Each one is deployed as its own Cloud Run service (see agents/service.py).
"""
from __future__ import annotations

from typing import Any

from loanflow.agents.react import BaseReActAgent, PlannedAction, raise_tool_errors
from loanflow.common.config import get_settings
from loanflow.common.schemas import (
    POLICY_VERSION,
    AssessRiskOutput,
    CheckComplianceOutput,
    CheckCreditOutput,
    FetchDocumentOutput,
    FinalDecision,
    ValidateDocumentOutput,
)
from loanflow.security.guardrails import contains_pii, ungrounded_numbers


def _ctx(task: dict[str, Any], node: str) -> dict[str, Any]:
    return (task.get("context") or {}).get(node) or {}


# =========================================================== 1. fetch_document
class FetchDocumentAgent(BaseReActAgent):
    name = "fetch_document"
    description = "Retrieve the applicant's loan document from the DMS"
    role_prompt = ("You are the Document Retrieval agent of a bank's loan-origination system. "
                   "Fetch exactly the requested document for the requested applicant.")
    tools = {"document": ["get_document"]}
    required_tools = ("get_document",)
    output_model = FetchDocumentOutput

    def plan(self, task, obs, scratch) -> PlannedAction:
        return self.next_in_sequence(
            [("get_document", "I need the document text before anything else can be checked.",
              {"applicant_id": task["applicant_id"], "document_id": task["document_id"]})], obs)

    def finalize(self, task, obs, scratch, notes):
        raise_tool_errors(obs)
        d = obs["get_document"]
        return {k: d[k] for k in ("document_id", "applicant_id", "doc_type", "content", "content_sha256",
                                  "pii_redacted", "injection_flags")}


# =========================================================== 2. validate_document
class ValidateDocumentAgent(BaseReActAgent):
    name = "validate_document"
    description = "Extract fields, check completeness and verify KYC"
    role_prompt = ("You are the Document Validation agent. Extract the application fields, check that all "
                   "mandatory fields are present and verify the applicant's KYC status.")
    tools = {"document": ["extract_application_fields", "verify_kyc"]}
    required_tools = ("extract_application_fields", "verify_kyc")
    output_model = ValidateDocumentOutput

    def plan(self, task, obs, scratch):
        ids = {"applicant_id": task["applicant_id"], "document_id": task["document_id"]}
        return self.next_in_sequence([
            ("extract_application_fields", "Extract structured fields from the document.", ids),
            ("verify_kyc", "Fields extracted; now verify PAN / CKYC / e-sign.", {"applicant_id": task["applicant_id"]}),
        ], obs)

    def finalize(self, task, obs, scratch, notes):
        raise_tool_errors(obs)
        ext = obs["extract_application_fields"]
        kyc = obs["verify_kyc"]
        fetched = _ctx(task, "fetch_document")
        issues: list[str] = []
        missing = list(ext["missing_fields"])
        if missing:
            issues.append(f"missing mandatory fields: {', '.join(missing)}")
        if not kyc["kyc_verified"]:
            failed = [k for k in ("pan_verified", "ckyc_found", "aadhaar_esign") if not kyc.get(k)]
            issues.append(f"KYC incomplete: {', '.join(failed)}")
        flags = fetched.get("injection_flags") or []
        if flags:
            issues.append(f"document contains suspected injected instructions ({', '.join(flags)})")
        extracted = None
        if not missing:
            f = ext["fields"]
            extracted = {
                "applicant_name": f["applicant_name"], "employment_type": f["employment_type"],
                "monthly_income_inr": f["monthly_income_inr"], "existing_emi_inr": f["existing_emi_inr"],
                "loan_amount_inr": f["loan_amount_inr"], "tenure_months": f["tenure_months"],
                "loan_type": f["loan_type"], "loan_purpose": f["loan_purpose"],
                "property_value_inr": f.get("property_value_inr"), "pan_last4": f.get("pan_last4"),
            }
        return {
            "is_valid": not missing and kyc["kyc_verified"],
            "extracted": extracted,
            "missing_fields": missing,
            "issues": issues,
            "kyc_verified": kyc["kyc_verified"],
            "requires_human_review": bool(flags),
        }


# =========================================================== 3. check_credit
class CheckCreditAgent(BaseReActAgent):
    name = "check_credit"
    description = "Pull the credit bureau report after confirming consent"
    role_prompt = ("You are the Credit Bureau agent. Confirm bureau consent first, then pull the CIBIL report "
                   "using the PAN last-4 from the validated application.")
    tools = {"credit_bureau": ["check_bureau_consent", "get_credit_report"]}
    required_tools = ("check_bureau_consent", "get_credit_report")
    output_model = CheckCreditOutput

    def plan(self, task, obs, scratch):
        ext = _ctx(task, "validate_document").get("extracted") or {}
        return self.next_in_sequence([
            ("check_bureau_consent", "CICRA requires consent before a bureau pull.", {"applicant_id": task["applicant_id"]}),
            ("get_credit_report", "Consent confirmed; pull the bureau report.",
             {"applicant_id": task["applicant_id"], "pan_last4": ext.get("pan_last4") or ""}),
        ], obs)

    def finalize(self, task, obs, scratch, notes):
        raise_tool_errors(obs)
        r = obs["get_credit_report"]
        return {k: r[k] for k in ("bureau", "credit_score", "active_loans", "defaults_last_24m", "enquiries_last_6m",
                                  "credit_utilisation_pct", "flags", "meets_minimum")}


# =========================================================== 4. assess_risk
class AssessRiskAgent(BaseReActAgent):
    name = "assess_risk"
    description = "Score credit risk, affordability (DTI/LTV) and pricing"
    role_prompt = ("You are the Risk Assessment agent. Get the product base rate, compute the EMI and run the "
                   "bank's risk-scoring model. Never estimate figures yourself - use the tools.")
    tools = {"risk_engine": ["get_base_rate", "calculate_emi", "compute_risk_score"]}
    required_tools = ("compute_risk_score",)
    output_model = AssessRiskOutput

    def plan(self, task, obs, scratch):
        ext = _ctx(task, "validate_document").get("extracted") or {}
        cr = _ctx(task, "check_credit")
        rate = (obs.get("get_base_rate") or {}).get("base_rate_pct", 12.0)
        return self.next_in_sequence([
            ("get_base_rate", "Need the product base rate for pricing.", {"loan_type": ext.get("loan_type", "")}),
            ("calculate_emi", "Compute indicative EMI at base rate.",
             {"principal_inr": ext.get("loan_amount_inr", 0), "annual_rate_pct": rate,
              "tenure_months": ext.get("tenure_months", 12)}),
            ("compute_risk_score", "Run the risk model with bureau and affordability inputs.", {
                "credit_score": cr.get("credit_score", 300), "monthly_income_inr": ext.get("monthly_income_inr", 0),
                "existing_emi_inr": ext.get("existing_emi_inr", 0), "loan_amount_inr": ext.get("loan_amount_inr", 0),
                "tenure_months": ext.get("tenure_months", 12), "loan_type": ext.get("loan_type", ""),
                "employment_type": ext.get("employment_type", "unknown"),
                "defaults_last_24m": cr.get("defaults_last_24m", 0), "enquiries_last_6m": cr.get("enquiries_last_6m", 0),
                "credit_utilisation_pct": cr.get("credit_utilisation_pct", 0.0),
                "property_value_inr": ext.get("property_value_inr"),
            }),
        ], obs)

    def finalize(self, task, obs, scratch, notes):
        raise_tool_errors(obs)
        r = obs["compute_risk_score"]
        return {
            "risk_grade": r["risk_grade"], "risk_score": r["risk_score"], "proposed_emi_inr": r["proposed_emi_inr"],
            "dti": r["dti"], "ltv": r["ltv"], "factors": r["factors"] + [f"HARD STOP: {h}" for h in r["hard_stops"]],
            "recommended_decision": r["recommended_decision"], "suggested_rate_pct": r["suggested_rate_pct"],
        }


# =========================================================== 5. check_compliance
class CheckComplianceAgent(BaseReActAgent):
    name = "check_compliance"
    description = "Sanctions, PEP, AML risk and RBI lending-norm checks"
    role_prompt = ("You are the Compliance agent (AML/CFT and RBI norms). Screen sanctions and PEP lists, "
                   "categorise AML risk and validate RBI lending rules. A sanctions hit is always non-compliant.")
    tools = {"compliance": ["screen_sanctions", "check_pep", "assess_aml_risk", "check_lending_regulations"]}
    required_tools = ("screen_sanctions", "check_pep", "assess_aml_risk", "check_lending_regulations")
    output_model = CheckComplianceOutput

    def plan(self, task, obs, scratch):
        ext = _ctx(task, "validate_document").get("extracted") or {}
        name = ext.get("applicant_name", "")
        return self.next_in_sequence([
            ("screen_sanctions", "Sanctions screening is mandatory under PMLA/UAPA.", {"applicant_name": name}),
            ("check_pep", "Check PEP status for enhanced due diligence.", {"applicant_name": name}),
            ("assess_aml_risk", "Categorise AML risk.", {
                "applicant_id": task["applicant_id"], "loan_amount_inr": ext.get("loan_amount_inr", 0),
                "monthly_income_inr": ext.get("monthly_income_inr", 0),
                "employment_type": ext.get("employment_type", "unknown"), "loan_purpose": ext.get("loan_purpose", "")}),
            ("check_lending_regulations", "Validate RBI lending norms.", {
                "loan_type": ext.get("loan_type", ""), "loan_amount_inr": ext.get("loan_amount_inr", 0),
                "tenure_months": ext.get("tenure_months", 12), "property_value_inr": ext.get("property_value_inr")}),
        ], obs)

    def finalize(self, task, obs, scratch, notes):
        raise_tool_errors(obs)
        sanc, pep, aml, regs = (obs[t] for t in self.required_tools)
        checks = [
            {"name": "sanctions_screening", "passed": not sanc["sanctions_hit"],
             "detail": f"best match score {sanc['match_score']}" + (f" on {sanc['matched_list']}" if sanc["sanctions_hit"] else ""),
             "regulation": sanc["regulation"]},
            {"name": "pep_screening", "passed": True, "detail": "PEP - EDD required" if pep["pep"] else "not a PEP",
             "regulation": pep["regulation"]},
            {"name": "aml_risk", "passed": aml["aml_risk"] != "high", "detail": f"{aml['aml_risk']}: {'; '.join(aml['reasons']) or 'no indicators'}",
             "regulation": aml["regulation"]},
            *regs["checks"],
        ]
        compliant = not sanc["sanctions_hit"] and regs["all_passed"]
        return {
            "compliant": compliant, "sanctions_hit": sanc["sanctions_hit"], "pep": pep["pep"], "aml_risk": aml["aml_risk"],
            "checks": checks, "requires_human_review": compliant and (pep["pep"] or aml["aml_risk"] == "high"),
        }


# =========================================================== 6/7. compile nodes
def _audit(task):
    return (task.get("context") or {}).get("_audit", [])


class _CompileBase(BaseReActAgent):
    tools = {"reporting": ["render_decision_report", "store_decision", "notify_applicant"]}
    required_tools = ("render_decision_report", "store_decision", "notify_applicant")
    output_model = FinalDecision
    memory_kinds = ("semantic", "procedural", "episodic")

    def draft(self, task) -> dict[str, Any]:
        raise NotImplementedError

    def prepare(self, task):
        return {"draft": self.draft(task)}

    def plan(self, task, obs, scratch):
        d = scratch["draft"]
        report = obs.get("render_decision_report") or {}
        return self.next_in_sequence([
            ("render_decision_report", "Render the decision report from the rule-engine results.", {"decision": d}),
            ("store_decision", "Persist the decision to the immutable audit store.",
             {"decision": {**d, "report_sha256": report.get("report_sha256")}}),
            ("notify_applicant", "Queue the customer notification.",
             {"applicant_id": d["applicant_id"], "decision_id": d["decision_id"], "decision": d["decision"]}),
        ], obs)

    def finalize(self, task, obs, scratch, notes):
        raise_tool_errors(obs)
        d = dict(scratch["draft"])
        facts = str(task.get("context"))
        if notes and not contains_pii(notes) and not ungrounded_numbers(notes, facts + str(d)):
            d["summary"] = notes  # grounded LLM narrative
        d["report_sha256"] = obs["render_decision_report"]["report_sha256"]
        d["audit_trail"] = [*_audit(task), {"actor": self.name, "action": "decision_stored",
                                            "detail": f"audit_id={obs['store_decision']['audit_id']} chain={obs['store_decision']['chain_hash'][:16]}"}]
        return d


class CompileReportAgent(_CompileBase):
    name = "compile_report"
    description = "Compile the approval decision, report and audit record"
    role_prompt = ("You are the Decision Compiler for APPROVED applications. Render, store and notify. "
                   "Your final_notes must restate only figures present in the facts.")

    def draft(self, task):
        c = task.get("context") or {}
        ext = c["validate_document"]["extracted"]
        risk, credit = c["assess_risk"], c["check_credit"]
        human = c.get("human_review")
        conditions = ["Key Fact Statement (KFS) to be issued and acknowledged before disbursal"]
        if ext["loan_type"] == "Home Loan":
            conditions.append("Equitable mortgage of the property and legal/technical valuation report")
        if human:
            conditions += human.get("conditions", [])
        reasons = [
            f"Credit score {credit['credit_score']} meets the policy minimum of {get_settings().min_credit_score}",
            f"Risk grade {risk['risk_grade']} (PD proxy {risk['risk_score']})",
            f"Debt-to-income {risk['dti']:.2f} within the {get_settings().max_dti:.2f} limit",
            "Sanctions, PEP, AML and RBI lending checks passed",
        ]
        if human:
            reasons.append(f"Approved by underwriter {human['reviewer']}: {human['notes']}")
        return {
            "decision_id": f"dec_{task['run_id'][-16:]}", "run_id": task["run_id"], "applicant_id": task["applicant_id"],
            "document_id": task["document_id"], "decision": "APPROVED", "reasons": reasons, "conditions": conditions,
            "approved_amount_inr": ext["loan_amount_inr"], "interest_rate_pct": risk["suggested_rate_pct"],
            "tenure_months": ext["tenure_months"], "risk_grade": risk["risk_grade"], "credit_score": credit["credit_score"],
            "summary": (f"{ext['loan_type']} of INR {ext['loan_amount_inr']:,.0f} approved at "
                        f"{risk['suggested_rate_pct']}% p.a. for {ext['tenure_months']} months "
                        f"(EMI about INR {risk['proposed_emi_inr']:,.0f})."),
            "human_review": human and {k: human[k] for k in ("reviewer", "decision", "notes")},
            "policy_version": POLICY_VERSION,
        }


class CompileRejectionAgent(_CompileBase):
    name = "compile_rejection"
    description = "Compile the rejection decision with policy reasons and audit record"
    role_prompt = ("You are the Decision Compiler for REJECTED applications. State the policy reasons factually. "
                   "Never disclose sanctions-list details in customer-facing text (PMLA tipping-off rule).")

    def draft(self, task):
        c = task.get("context") or {}
        reasons: list[str] = []
        if err := c.get("_error"):
            reasons.append(f"Application could not be completed: {err.get('code')} - {err.get('message')}")
        v = c.get("validate_document") or {}
        if v and not v.get("is_valid"):
            reasons += [f"Document validation failed: {i}" for i in v.get("issues", [])] or ["Document validation failed"]
        cr = c.get("check_credit") or {}
        if cr and not cr.get("meets_minimum"):
            reasons.append(f"Credit score {cr['credit_score']} / bureau history does not meet policy")
            reasons += [f"Bureau flag: {f}" for f in cr.get("flags", [])]
        r = c.get("assess_risk") or {}
        if r and r.get("recommended_decision") == "REJECTED":
            reasons += [f.removeprefix("HARD STOP: ") for f in r.get("factors", []) if f.startswith("HARD STOP")]
            if r.get("risk_grade") in ("D", "E"):
                reasons.append(f"Risk grade {r['risk_grade']} outside appetite")
        comp = c.get("check_compliance") or {}
        if comp and not comp.get("compliant"):
            reasons += [f"Compliance check failed: {ch['name']}" for ch in comp.get("checks", []) if not ch["passed"]]
        human = c.get("human_review")
        if human and human.get("decision") == "REJECTED":
            reasons.append(f"Declined by underwriter {human['reviewer']}: {human['notes']}")
        reasons = list(dict.fromkeys(reasons)) or ["Application does not meet the bank's lending policy"]
        return {
            "decision_id": f"dec_{task['run_id'][-16:]}", "run_id": task["run_id"], "applicant_id": task["applicant_id"],
            "document_id": task["document_id"], "decision": "REJECTED", "reasons": reasons, "conditions": [],
            "approved_amount_inr": None, "interest_rate_pct": None, "tenure_months": None,
            "risk_grade": r.get("risk_grade"), "credit_score": cr.get("credit_score"),
            "summary": "We are unable to approve this application as it does not meet the bank's current lending policy.",
            "human_review": human and {k: human[k] for k in ("reviewer", "decision", "notes")},
            "policy_version": POLICY_VERSION,
        }


AGENTS: dict[str, BaseReActAgent] = {
    a.name: a for a in (
        FetchDocumentAgent(), ValidateDocumentAgent(), CheckCreditAgent(), AssessRiskAgent(),
        CheckComplianceAgent(), CompileReportAgent(), CompileRejectionAgent(),
    )
}
