import os
import cv2
import streamlit as st
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="مكتشف نماذج الشارتات", layout="wide")
st.title("📈 نظام مطابقة وتحديد شارتات العملات الرقمية")
st.write("قم برفع صورة الشارت الحالية، وسيقوم النظام بالبحث في مخزن الصور لديك عن الشارتات الأكثر شبهاً.")

# تحديد مجلد الصور (قاعدة البيانات المحلية الخاصة بك)
# تأكد من إنشاء هذا المجلد ووضع صورك داخله
IMAGE_DATABASE_DIR = "chart_database"

def calculate_similarity(img1_path, img2_path):
    """حساب نسبة الشبه بين صورتين باستخدام خوارزمية ORB والمطابقة الذكية"""
    # قراءة الصور بالأبيض والأسود لتسريع وتحسين المقارنة بناءً على الشكل فقط
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0.0
    
    # استخدام مستشعر الميزات ORB (سريع وفعال للشارتات)
    orb = cv2.ORB_create(nfeatures=1000)
    
    # إيجاد النقاط المفتاحية ووصفها
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0.0
        
    # مطابقة النقاط بين الصورتين
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # ترتيب المطابقات حسب المسافة (الأقرب يعني الأكثر شبهاً)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # حساب نسبة تقريبية للشبه بناءً على عدد النقاط المتطابقة الجيدة
    good_matches = [m for m in matches if m.distance < 50]
    
    if len(matches) == 0:
        return 0.0
    
    similarity = (len(good_matches) / max(len(kp1), len(kp2))) * 100
    return min(similarity * 4, 100.0) # تعديل النسبة لتكون منطقية للعرض

# التأكد من وجود مجلد الصور
if not os.path.exists(IMAGE_DATABASE_DIR):
    os.makedirs(IMAGE_DATABASE_DIR)
    st.warning(f"تم إنشاء مجلد جديد باسم '{IMAGE_DATABASE_DIR}'. يرجى وضع صور الشارتات الخاصة بك داخله ثم تحديث الصفحة.")

# واجهة رفع الصورة الجديدة
uploaded_file = st.file_uploader("اختر صورة الشارت التي تريد فحصها...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # حفظ الصورة المرفوعة مؤقتاً لمقارنتها
    temp_input_path = "temp_query_chart.png"
    with open(temp_input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.subheader("🔍 الشارت المطلوب فدحه:")
    st.image(uploaded_file, width=400)
    
    # زر بدء البحث
    if st.button("ابدأ البحث عن الشارتات المطابقة"):
        results = []
        
        # قراءة المجلد والمقارنة مع كل الصور
        all_images = [f for f in os.listdir(IMAGE_DATABASE_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        if len(all_images) == 0:
            st.error("مجلد الصور فارغ! ضع بعض الصور في مجلد 'chart_database' أولاً.")
        else:
            with st.spinner("جاري فحص وتدقيق قاعدة البيانات والبحث عن التطابقات..."):
                for img_name in all_images:
                    db_img_path = os.path.join(IMAGE_DATABASE_DIR, img_name)
                    # حساب الشبه
                    score = calculate_similarity(temp_input_path, db_img_path)
                    if score > 5: # تصفية النتائج غير الشبيهة تماماً
                        results.append((img_name, score))
            
            # ترتيب النتائج من الأعلى شبهاً إلى الأقل
            results = sorted(results, key=lambda x: x[1], reverse=True)
            
            # عرض أفضل 3 نتائج
            st.success("اكتمل البحث! إليك الشارتات الأكثر شبهاً في قاعدة بياناتك:")
            
            top_results = results[:3] # خذ أعلى 3 شارتات متشابهة
            
            if not top_results:
                st.info("لم يتم العثور على شارتات ذات نسبة شبه عالية.")
            else:
                cols = st.columns(len(top_results))
                for idx, (img_name, score) in enumerate(top_results):
                    with cols[idx]:
                        st.metric(label="نسبة التطابق التقريبية", value=f"{score:.1f}%")
                        full_path = os.path.join(IMAGE_DATABASE_DIR, img_name)
                        st.image(Image.open(full_path), caption=f"اسم الملف: {img_name}")
