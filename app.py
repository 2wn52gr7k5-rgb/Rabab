import os
import cv2
import streamlit as st
from PIL import Image

# إعداد واجهة التطبيق
st.set_page_config(page_title="نماذج الشارتات", layout="wide")
st.title("📈 نظام مطابقة وتحديد شارتات العملات الرقمية")
st.write("قم برفع صورة الشارت الحالية، وسيقوم النظام بالبحث في مخزن الصور لديك عن الشارتات الأكثر شبهاً.")

# مجلد الصور (قاعدة البيانات المحلية الخاصة بك)
IMAGE_DATABASE_DIR = "chart_database"

def calculate_similarity(img1_path, img2_path):
    """حساب نسبة الشبه بين صورتين باستخدام خوارزمية ORB"""
    # قراءة الصور مع تفعيل إجبار القراءة لتفادي تعليق السيرفر
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        return 0.0
        
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return 0.0
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    if not matches:
        return 0.0
        
    similar_score = (len(matches) / max(len(kp1), len(kp2))) * 100
    return round(similar_score, 1)

# رفع الصورة من المستخدم
uploaded_file = st.file_uploader("...اختر صورة الشارت التي تريد فحصها", type=["jpg", "png", "jpeg", "PNG", "JPG", "JPEG"])

if uploaded_file is not None:
    temp_uploaded_path = "temp_uploaded_chart.png"
    with open(temp_uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.subheader("🔍 الشارت المطلوب فحصه:")
    st.image(uploaded_file, use_container_width=True)
    
    if st.button("ابدأ البحث عن الشارتات المطابقة"):
        if not os.path.exists(IMAGE_DATABASE_DIR) or len(os.listdir(IMAGE_DATABASE_DIR)) == 0:
            st.error(f"مجلد الصور '{IMAGE_DATABASE_DIR}' فارغ! ضع بعض الصور في المجلد أولاً.")
        else:
            all_results = []
            
            # جلب قائمة الصور وعمل تحديث حقيقي للمسارات في كل ضغطة زر
            files_list = [f for f in os.listdir(IMAGE_DATABASE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            for file_name in files_list:
                db_img_path = os.path.join(IMAGE_DATABASE_DIR, file_name)
                # حساب النسبة بشكل مستقل لكل صورة
                score = calculate_similarity(temp_uploaded_path, db_img_path)
                all_results.append({"file_name": file_name, "score": score, "path": db_img_path})
            
            if not all_results:
                st.warning("لم يتم العثور على صور مدعومة داخل المجلد.")
            else:
                # ترتيب كل نتائج الصور المخزنة من الأعلى للأقل
                all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)
                
                st.success("🎉 اكتمل البحث بنجاح في كامل المخزن!")
                
                # عرض أفضل 3 نتائج مختلفة بجانب بعضها لضمان التنقل بين الصور
                top_results = all_results[:3]
                cols = st.columns(len(top_results))
                
                for index, res in enumerate(top_results):
                    with cols[index]:
                        st.metric(label=f"المركز {index+1} - التطابق", value=f"{res['score']}%")
                        img = Image.open(res['path'])
                        st.image(img, caption=res['file_name'], use_container_width=True)
                        
    if os.path.exists(temp_uploaded_path):
        os.remove(temp_uploaded_path)
