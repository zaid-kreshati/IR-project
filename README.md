# 📚 مشروع استرجاع المعلومات - 2024/2025

نظام استرجاع معلومات (Information Retrieval System) متكامل مبني باستخدام FastAPI وReact، يدعم تمثيل المستندات بعدة طرق مثل TF-IDF وBERT، ويعتمد على بنية خدمية SOA (Service-Oriented Architecture). يستخدم MongoDB كقاعدة بيانات، ويوفر واجهة أمامية لعرض نتائج البحث وتقييمها.

---

## 🎯 هدف المشروع

تطوير نظام قادر على استرجاع المعلومات من قواعد بيانات نصية ضخمة باستخدام استعلامات بلغة طبيعية، مع دعم تقييم الأداء باستخدام مقاييس IR القياسية.

---

## 🧠 المزايا الأساسية

- ✅ دعم أكثر من مجموعة بيانات (datasets) مثل: `antique`, `arguana`, `fiqa`, `climate-fever`
- ✅ اختيار تمثيل المستندات: `TF-IDF`, `BERT`, أو `Hybrid`
- ✅ تصحيح الإملاء وتوسيع الاستعلامات واقتراحات تلقائية
- ✅ عرض النتائج بترتيب الصلة مع النصوص
- ✅ تقييم الأداء عبر:
  - MAP (Mean Average Precision)
  - MRR (Mean Reciprocal Rank)
  - Precision@10
  - Recall

---

## 🧰 التقنيات المستخدمة

| المجال         | التقنية                            |
| -------------- | ---------------------------------- |
| Backend        | FastAPI, Python, joblib            |
| Frontend       | React, TailwindCSS                 |
| قاعدة البيانات | MongoDB                            |
| تمثيل النصوص   | TF-IDF, Sentence-BERT (MiniLM)     |
| التقييم        | ملفات qrels/queries من ir-datasets |

---

## 🏗️ هيكل المشروع

ir-project/
├── backend/
│ ├── main.py
│ ├── services/
│ ├── utils/
│ └── models/
├── frontend/
│ ├── src/
│ └── pages/
├── data/
│ └── indexed_documents/
├── scripts/
│ ├── ingest_datasets.py
│ └── evaluate.py
├── README.md
└── requirements.txt

---

## 🚀 خطوات التشغيل

### 1. تشغيل MongoDB:

احرص على أن MongoDB يعمل محليًا على `localhost:27017`.

### 2. تشغيل الواجهة الخلفية (backend):

cd backend
pip install -r requirements.txt
uvicorn main:app --reload.

### 3. تشغيل الواجهة الأمامية (frontend):
cd frontend
npm install
npm run dev

 :طريقة التقييم

النظام يستخدم ملفات qrels وqueries لتقييم نتائج البحث. يتم احتساب:

MAP: دقة النتائج الإجمالية
MRR: ترتيب أول نتيجة صحيحة
Precision@10: دقة أول 10 نتائج
Recall: قدرة النظام على استرجاع كل النتائج المهمة


