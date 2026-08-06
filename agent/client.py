import asyncio
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ==========================================
# 1. معالجات الأحداث (Event & Callback Handlers)
# ==========================================

def progress_callback(progress: float, total: float | None):
    """دالة استقبال تحديثات شريط التقدم من الخادم (Progress Tracking)"""
    percentage = (progress / total) * 100 if total else progress
    print(f"⏳ [إشعار تقدم من الخادم]: اكتمل {percentage:.0f}% من العملية...")

def handle_list_changed_notification():
    """دالة استقبال إشعار تغيير قائمة الأدوات الديناميكية (Notifications)"""
    print("\n🔔 [إشعار بروتوكول]: تم تلقي notifications/tools/list_changed! تغيرت صلاحيات/أدوات الخادم.")

# ==========================================
# 2. المشغل الرئيسي للعميل (Agent Execution Loop)
# ==========================================

async def run_harborstone_agent():
    print("🚀 بدء تشغيل عميل Harborstone Insurance MCP Agent...\n")

    # أ) تحديد بارامترات الاتصال بخادم MCP عبر Stdio
    server_params = StdioServerParameters(
        command=sys.executable, # يضمن هذا السطر استخدام بايثون الخاص بـ venv حصراً
        args=["mcp_server/server.py"], 
        env=None
    )

    # ب) فتح قناة الاتصال بإنبوب Stdio
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            
            # ----------------------------------------------------
            # Step 1: المصافحة والتحقق من الصلاحيات (Capability Negotiation)
            # ----------------------------------------------------
            print("--- 1. المصافحة وإعلان الصلاحيات (Handshake) ---")
            init_result = await session.initialize()
            print(f"✅ تم الاتصال بنجاح مع الخادم: {init_result.serverInfo.name}")
            
            # فحص الصلاحيات المُعلنة من الخادم قبل الاعتماد عليها
            capabilities = init_result.capabilities
            print(f"📋 الصلاحيات المتاحة: Tools={bool(capabilities.tools)}, Resources={bool(capabilities.resources)}, Prompts={bool(capabilities.prompts)}\n")

            # ----------------------------------------------------
            # Step 2: قراءة الموارد (Resources/Read)
            # ----------------------------------------------------
            print("--- 2. قراءة الموارد الاستاتيكية والديناميكية (Resources) ---")
            
            # قراءة المورد 1: شروط وأحكام وثيقة التأمين
            terms_resource = await session.read_resource("harborstone://policies/terms/marine-hull")
            print("📜 [المورد 1 - شروط وثيقة التأمين البحري]:")
            print(terms_resource.contents[0].text)

            # قراءة المورد 2: تقرير المطالبات المعلقة
            pending_claims = await session.read_resource("harborstone://claims/pending-summary")
            print("\n📊 [المورد 2 - سجل المطالبات المعلقة في الداتابيز]:")
            print(pending_claims.contents[0].text)
            print()

            # ----------------------------------------------------
            # Step 3: استخدام قوالب الأوامر الجاهزة (MCP Prompts)
            # ----------------------------------------------------
            print("--- 3. جلب واستخدام قوالب المطالبات الجاهزة (Prompts) ---")
            prompt_data = await session.get_prompt("draft_claim_investigation", arguments={"claim_id": "3"})
            print("💡 [القالب المولد للتحقيق في المطالبة رقم 3]:")
            print(prompt_data.messages[0].content.text)
            print()

            # ----------------------------------------------------
            # Step 4: استدعاء أداة طويلة مع تتبع التقدم (Progress Tracking)
            # ----------------------------------------------------
            print("--- 4. تنفيذ أداة تدقيق المخاطر (Progress Tracking) ---")
            # تم إزالة المعامل on_progress ليتوافق مع مكتبة العميل الحالية
            audit_response = await session.call_tool(
                "audit_high_risk_policies",
                arguments={}
            )
            print(f"🎯 [نتيجة الأداة]: {audit_response.content[0].text}\n")

            # ----------------------------------------------------
            # Step 5: اختبار الحماية والتأكيد البشري (Auth & Elicitation)
            # ----------------------------------------------------
            print("--- 5. اختبار الأمان والتأكيد البشري (Authorization & Elicitation) ---")
            
            # محاولة 1: رفض الصلاحية للـ Customer Support
            print("🔒 [محاولة 1]: طلب موافقة بصلاحية 'Customer Support'...")
            unauth_res = await session.call_tool(
                "approve_claim",
                arguments={"claim_id": 3, "user_role": "Customer Support"}
            )
            print(f"🛡️ [رد السيرفر]: {unauth_res.content[0].text}\n")

            # تغيير الدور ودفع الإشعار
            print("🔄 [تغيير الدور]: رفع الصلاحية إلى 'Claims Officer'...")
            role_res = await session.call_tool(
                "switch_user_role",
                arguments={"new_role": "Claims Officer"}
            )
            print(f"📣 [رد السيرفر]: {role_res.content[0].text}")
            handle_list_changed_notification()
            print()

            # محاولة 2: الموافقة بصلاحية عالية ومبلغ ضخم ($38,000) لتفعيل الـ Elicitation
            print("⚠️ [محاولة 2]: طلب موافقة على المطالبة رقم 3 بقيمة تتجاوز الحد الآمن...")
            
            # محاكاة التوقف والتأكيد البشري (Mid-call Elicitation)
            print("\n" + "="*50)
            print("🛑 [تنبيه Elicitation من السيرفر]: المطالبة رقم 3 قيمتها $38,000.00.")
            user_input = input("هل تؤكد الموافقة على صرف المبلغ من قاعدة البيانات؟ (yes/no): ").strip().lower()
            print("="*50 + "\n")

            if user_input in ["yes", "y", "نعم"]:
                approve_res = await session.call_tool(
                    "approve_claim",
                    arguments={"claim_id": 3, "user_role": "Claims Officer"}
                )
                print(f"🎉 [النتيجة النهائية]: {approve_res.content[0].text}")
            else:
                print("❌ تم إلغاء العملية بناءً على قرار المستخدم البشري.")

if __name__ == "__main__":
    asyncio.run(run_harborstone_agent())