import asyncio
import json
import mysql.connector
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# الاعتماد على الحزمة المستقلة المتوافقة
from fastmcp import FastMCP, Context

# ==========================================
# 1. الاتصال بقاعدة البيانات (Harborstone Insurance DB)
# ==========================================
def get_db_connection():
    """إنشاء اتصال مباشر بقاعدة بيانات XAMPP MySQL"""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="", # كلمة السر الافتراضية في XAMPP
        database="harborstone_insurance"
    )

# ==========================================
# 2. تهيئة خادم MCP والإعلان عن الصلاحيات (Capability Negotiation)
# ==========================================
# إزالة معامل dependencies ليتوافق مع الحزمة المستقلة
mcp = FastMCP(
    name="Harborstone-Marine-Insurance-Server"
)

# حالة الجلسة للتنقل بين الأدوار (Role-Based State)
CURRENT_USER_ROLE = "Customer Support" # الدور الافتراضي عند بدء الاتصال


# ==========================================
# 3. التصميم الدفاعي للنماذج (Defensive Tool Design & Pydantic Validation)
# ==========================================
class SubmitClaimInput(BaseModel):
    """مخطط مدخلات تقديم المطالبة مع شروط تحقق صارمة"""
    policy_id: int = Field(..., gt=0, description="معرف الوثيقة ويجب أن يكون رقماً موجباً أكبر من 0")
    description: str = Field(..., min_length=10, max_length=500, description="وصف حادث السفينة بالتفصيل")
    amount: float = Field(..., gt=0.0, description="قيمة التعويض المطلوبة بالدولار")
    
    class Config:
        extra = "forbid" # يطابق additionalProperties: false لحظر أي حقول غير معتمدة


# ==========================================
# 4. الموارد (Resources) - بيانات للقراءة فقط
# ==========================================
@mcp.resource("harborstone://policies/terms/marine-hull")
def get_marine_hull_terms() -> str:
    """المورد 1: وثيقة شروط وأحكام التأمين البحري على هيكل السفينة (Read-Only Policy)"""
    return """
    === HARBORSTONE INSURANCE: MARINE HULL POLICY TERMS ===
    1. Claims above $10,000 USD require explicit approval from a Senior Claims Officer.
    2. Storm damage claims must include official meteorological verification.
    3. Engine failures are subject to proof of regular maintenance within 6 months.
    """

@mcp.resource("harborstone://claims/pending-summary")
def get_pending_claims_summary() -> str:
    """المورد 2: تقرير مباشر عن كافة المطالبات المعلقة في النظام"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT claim_id, policy_id, amount, description FROM Claims WHERE status = 'Pending'")
    claims = cursor.fetchall()
    conn.close()
    return json.dumps(claims, indent=2, default=str)


# ==========================================
# 5. قوالب المطالبات الجاهزة (Prompts)
# ==========================================
@mcp.prompt()
def draft_claim_investigation(claim_id: int) -> str:
    """قالب مطالبة استباقي لتوجيه النموذج في تحليل ملف المطالبة"""
    return f"""قم بمراجعة المطالبة رقم {claim_id} من قاعدة بيانات Harborstone Insurance.
1. استعلم عن تفاصيل المطالبة ورقم الوثيقة والسفينة المرتبطة بها.
2. قارن مبلغ المطالبة بشروط وثيقة التأمين المتاحة في المورد: harborstone://policies/terms/marine-hull
3. قدم توصية فنية ملخصة للعميل أو ضابط المطالبات حول قبول أو رفض الطلب."""


# ==========================================
# 6. الأدوات وتتبع التقدم (Tools & Progress Tracking)
# ==========================================
@mcp.tool()
async def audit_high_risk_policies(ctx: Context) -> str:
    """أداة تدقيق الوثائق عالية المخاطر ذات التغطيات المالية الضخمة مع إرسال تقارير التقدم"""
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # 1. إشعار بدء العملية (20%)
    await ctx.report_progress(progress=20, total=100)
    cursor.execute("SELECT policy_id, premium, status FROM Policies WHERE premium > 8000.00")
    policies = cursor.fetchall()
    
    # 2. إشعار معالجة البيانات (60%)
    await ctx.report_progress(progress=60, total=100)
    await asyncio.sleep(1) # محاكاة عملية معالجة وحسابات معقدة
    
    high_risk_count = len(policies)
    
    # 3. إشعار الاكتفاء والأكتمال (100%)
    await ctx.report_progress(progress=100, total=100)
    conn.close()
    
    return f"تم فحص {high_risk_count} وثيقة عالية المخاطر بنجاح. القسط التراكمي للوثائق يتجاوز الحد الآمن."


# ==========================================
# 7. Planning integration tools (read-only underwriting lookups)
# ==========================================
@mcp.tool()
def get_customer_policies(customer_id: int) -> str:
    """Return the customer's active/recent marine policies from the real DB."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """SELECT policy_id, customer_id, vessel_id, policy_type, start_date,
                      end_date, premium, status
               FROM Policies WHERE customer_id = %s
               ORDER BY policy_id""",
            (customer_id,),
        )
        return json.dumps(cursor.fetchall(), default=str)
    finally:
        conn.close()


@mcp.tool()
def get_vessel(vessel_id: int) -> str:
    """Return one vessel record from the real Harborstone DB."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM Vessels WHERE vessel_id = %s", (vessel_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Vessel {vessel_id} was not found")
        return json.dumps(row, default=str)
    finally:
        conn.close()


@mcp.tool()
def check_vessel_eligibility(
    vessel_type: str,
    year_built: int,
    value: float,
) -> str:
    """Apply Harborstone's fictional demo underwriting rules to a new vessel.

    Rules used only for this lab's reproducible evaluation:
    - vessel age must be <= 20 years;
    - declared value must be > 0;
    - supported types are Boat and Yacht.
    """
    from datetime import date

    current_year = date.today().year
    age = current_year - year_built
    reasons: list[str] = []
    if vessel_type not in {"Boat", "Yacht"}:
        reasons.append("Unsupported vessel type")
    if value <= 0:
        reasons.append("Declared value must be positive")
    if age > 20:
        reasons.append("Vessel is older than the 20-year underwriting limit")
    eligible = not reasons
    return json.dumps(
        {
            "eligible": eligible,
            "vessel_age": age,
            "reasons": reasons,
            "checked_rules": [
                "age <= 20 years",
                "type in {Boat, Yacht}",
                "value > 0",
            ],
        }
    )


@mcp.tool()
def estimate_policy_premium_change(
    current_premium: float,
    vessel_type: str,
    vessel_value: float,
) -> str:
    """Estimate the incremental annual premium using lab underwriting rates."""
    rates = {"Boat": 0.01, "Yacht": 0.015}
    if vessel_type not in rates:
        raise ValueError("Unsupported vessel type")
    if current_premium < 0 or vessel_value <= 0:
        raise ValueError("Premium and vessel value must be valid positive values")
    additional = round(vessel_value * rates[vessel_type], 2)
    return json.dumps(
        {
            "current_premium": round(current_premium, 2),
            "estimated_additional_premium": additional,
            "estimated_new_premium": round(current_premium + additional, 2),
            "rate_used": rates[vessel_type],
            "note": "Estimate for the Week 4 lab; final underwriting approval is required.",
        }
    )


@mcp.tool()
def get_policy_update_requirements(
    vessel_type: str,
    vessel_value: float,
) -> str:
    """Return deterministic documentation requirements for a policy update."""
    documents = ["Proof of ownership/purchase invoice", "Current vessel registration"]
    if vessel_type == "Yacht":
        documents.append("Current vessel valuation")
    if vessel_value >= 500000:
        documents.append("Recent independent valuation report")
    return json.dumps({"required_documents": documents})


# ==========================================
# 7. الأمان والتأكيد البشري (Defensive Logic & Elicitation)
# ==========================================
@mcp.tool()
async def approve_claim(claim_id: int, user_role: str, ctx: Context) -> str:
    """أداة الموافقة على المطالبة المالية مع التحقق من الصلاحيات والتأكيد البشري (Elicitation)"""
    
    # أ) Handler-Level Authorization (التحقق من الصلاحيات)
    if user_role not in ["Claims Officer", "Manager"]:
        return "خطأ أمني: لا تملك الصلاحية للقيام بالموافقة على المطالبات. يجب أن تكون Claims Officer أو Manager."

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT claim_id, amount, status FROM Claims WHERE claim_id = %s", (claim_id,))
    claim = cursor.fetchone()
    
    if not claim:
        conn.close()
        return f"خطأ: المطالبة رقم {claim_id} غير موجودة."
        
    if claim['status'] != 'Pending':
        conn.close()
        return f"تنبيه: المطالبة رقم {claim_id} ليست في حالة معلقة (الحالة الحالية: {claim['status']})."

    # ب) Elicitation (إيقاف التنفيذ مؤقتاً لطلب موافقة بشرية صريحة إذا كان المبلغ ضخماً)
    claim_amount = float(claim['amount'])
    if claim_amount > 10000.0:
        # إرسال طلب Elicitation للمستخدم البشري للموافقة صراحة قبل تغيير حالة الداتابيز
        confirmed = await ctx.session.create_elicitation(
            message=f"المطالبة رقم {claim_id} قيمتها عالية جداً (${claim_amount:,.2f}). هل تؤكد الموافقة على صرف هذا المبلغ؟"
        )
        if not confirmed:
            conn.close()
            return f"تم إلغاء عملية الموافقة على المطالبة {claim_id} بناءً على رفض المستخدم البشري."

    # ج) تنفيذ التغيير في قاعدة البيانات عند اجتياز كافة القيود
    cursor.execute("UPDATE Claims SET status = 'Approved' WHERE claim_id = %s", (claim_id,))
    conn.commit()
    conn.close()
    
    return f"تمت الموافقة بنجاح على المطالبة رقم {claim_id} بقيمة ${claim_amount:,.2f}."


# ==========================================
# 8. التحديثات الديناميكية وتغيير الأدوار (Notifications)
# ==========================================
@mcp.tool()
async def switch_user_role(new_role: str, ctx: Context) -> str:
    """تغيير دور المستخدم الحالي ودفع إشعار أدوات ديناميكي (notifications/tools/list_changed)"""
    global CURRENT_USER_ROLE
    CURRENT_USER_ROLE = new_role
    
    # إرسال إشعار للـ Client بحدوث تغيير في قائمة/صلاحيات الأدوات المتاحة
    await ctx.session.send_tool_list_changed()
    
    return f"تم تغيير دور الجلسة الحالية إلى: {new_role}. تم دفع إشعار tools/list_changed للعميل."


# ==========================================
# 9. تشغيل السيرفر (Stdio & Streamable HTTP)
# ==========================================
if __name__ == "__main__":
    # يعمل السيرفر افتراضياً عبر Stdio للتطوير المحلي، ويمكن نقله إلى Streamable HTTP للتطبيقات السحابية
    mcp.run(transport="stdio")