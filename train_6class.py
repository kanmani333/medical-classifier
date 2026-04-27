import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Load mtsamples ─────────────────────────────────────────────
df_mt = pd.read_csv('mtsamples.csv')
df_mt = df_mt.dropna(subset=['transcription'])
df_mt['text'] = (
    df_mt['sample_name'].fillna('') + ' ' +
    df_mt['description'].fillna('') + ' ' +
    df_mt['transcription'].fillna('')
)

categories = [
    'Radiology', 'Surgery', 'Neurology',
    'Gastroenterology', 'Cardiovascular / Pulmonary', 'General Medicine'
]

df_mt['label'] = df_mt['medical_specialty'].str.strip()
df_mt = df_mt[df_mt['label'].isin(categories)][['text', 'label']]
df_mt = df_mt[df_mt['text'].str.strip().str.len() > 50]

print("mtsamples counts:")
print(df_mt['label'].value_counts())

# ── Synthetic samples ──────────────────────────────────────────
synthetic = {
    'Neurology': [
        "Patient presents with seizures epilepsy EEG shows abnormal brain wave activity started antiepileptic medication neurologist evaluated focal deficit",
        "Parkinson disease tremor bradykinesia rigidity started levodopa carbidopa dopamine deficiency basal ganglia neurological examination",
        "Multiple sclerosis demyelination MRI brain white matter lesions optic neuritis weakness fatigue started interferon beta",
        "Stroke ischemic cerebrovascular accident left sided weakness aphasia CT brain MRI diffusion weighted imaging thrombolysis tPA",
        "Migraine headache photophobia phonophobia nausea vomiting aura neurologist prescribed sumatriptan prophylaxis topiramate",
        "Alzheimer dementia memory loss cognitive decline Mini Mental State Examination MMSE cholinesterase inhibitor donepezil",
        "Guillain Barre syndrome ascending paralysis areflexia nerve conduction study lumbar puncture CSF albuminocytologic dissociation",
        "Epilepsy seizure disorder EEG electroencephalogram brain activity anticonvulsant valproate phenytoin carbamazepine",
        "Cerebral palsy spasticity motor dysfunction physiotherapy botulinum toxin injection baclofen muscle relaxant",
        "Meningitis bacterial lumbar puncture CSF pleocytosis fever neck stiffness Kernig sign antibiotic ceftriaxone",
        "Neuropathy peripheral nerve conduction velocity electromyography EMG diabetic neuropathy glove stocking sensory loss",
        "Brain tumor glioblastoma astrocytoma MRI contrast enhancement mass effect edema surgical resection chemotherapy radiation",
        "Transient ischemic attack TIA carotid Doppler CT brain antiplatelet aspirin clopidogrel stroke prevention",
        "Myasthenia gravis acetylcholine receptor antibody ptosis diplopia edrophonium test pyridostigmine thymectomy",
        "Huntington disease chorea genetic CAG repeat basal ganglia neurodegeneration psychiatric symptoms",
    ],
    'Gastroenterology': [
        "Colonoscopy performed colon polyp removed biopsy taken patient had abdominal pain diarrhea irritable bowel syndrome",
        "Upper GI endoscopy gastric ulcer peptic ulcer disease Helicobacter pylori proton pump inhibitor omeprazole",
        "Liver cirrhosis hepatitis B hepatitis C jaundice ascites hepatic encephalopathy spironolactone lactulose",
        "Crohn disease inflammatory bowel disease terminal ileum stricture fistula corticosteroid azathioprine biologics infliximab",
        "Ulcerative colitis bloody diarrhea rectal bleeding colonoscopy biopsy mesalamine prednisolone colectomy",
        "Pancreatitis amylase lipase elevated CT abdomen pancreatic necrosis pseudocyst NPO bowel rest IV fluids",
        "Cholecystitis gallbladder stones cholelithiasis ERCP bile duct endoscopic sphincterotomy",
        "GERD gastroesophageal reflux disease esophagitis heartburn Barrett esophagus endoscopy proton pump inhibitor",
        "Celiac disease gluten intolerance villous atrophy duodenal biopsy tissue transglutaminase antibody gluten free diet",
        "Colorectal cancer CEA colonoscopy biopsy chemotherapy FOLFOX bevacizumab hepatic metastasis",
        "Acute gastroenteritis vomiting diarrhea dehydration oral rehydration stool culture rotavirus norovirus",
        "Hepatocellular carcinoma AFP alpha fetoprotein liver ultrasound TACE radiofrequency ablation sorafenib",
        "Primary biliary cirrhosis AMA antimitochondrial antibody ursodeoxycholic acid cholestasis pruritus",
        "Esophageal varices portal hypertension band ligation sclerotherapy propranolol TIPS bleeding",
        "Malabsorption steatorrhea fat soluble vitamins pancreatic exocrine insufficiency enzyme replacement",
    ],
    'Cardiovascular / Pulmonary': [
        "Myocardial infarction ST elevation ECG troponin elevated cardiac catheterization coronary artery disease stent angioplasty",
        "Heart failure ejection fraction echocardiogram BNP elevated furosemide ACE inhibitor carvedilol beta blocker",
        "Atrial fibrillation irregular rhythm anticoagulation warfarin rate control digoxin cardioversion ablation",
        "Hypertension blood pressure elevated amlodipine lisinopril hydrochlorothiazide target organ damage",
        "Pulmonary embolism CT pulmonary angiography DVT deep vein thrombosis anticoagulation heparin warfarin",
        "COPD chronic obstructive pulmonary disease emphysema bronchitis spirometry FEV1 salbutamol tiotropium",
        "Asthma bronchospasm wheeze inhaler salbutamol corticosteroid peak flow spirometry bronchodilator",
        "Pneumonia chest X-ray consolidation fever cough sputum antibiotic amoxicillin azithromycin",
        "Aortic stenosis murmur echocardiogram valve replacement cardiac surgery TAVR",
        "Ventricular tachycardia arrhythmia defibrillator ICD amiodarone electrophysiology study",
        "Pericarditis chest pain pericardial effusion echocardiogram pericardiocentesis colchicine NSAIDs",
        "Pulmonary hypertension right heart catheterization sildenafil bosentan prostacyclin right ventricular failure",
        "Aortic dissection chest pain CT angiography Stanford type A B surgical repair endovascular stent",
        "Cardiomyopathy dilated hypertrophic restrictive echocardiogram heart failure ICD transplant",
        "Pleural effusion thoracentesis Light criteria exudate transudate chest drain pleural biopsy",
    ],
    'Radiology': [
        "X-ray radiograph PA lateral view opacity consolidation cardiomegaly imaging report impression",
        "MRI T1 T2 FLAIR sequence gadolinium contrast signal intensity hypointense hyperintense",
        "CT scan hounsfield units hyperdense hypodense attenuation axial coronal sagittal reconstruction",
        "Ultrasound sonography echogenicity anechoic hyperechoic hypoechoic probe transducer acoustic",
        "PET scan FDG uptake standardized uptake value SUV tracer radiotracer nuclear imaging",
        "Mammography BI-RADS microcalcification spiculated mass lesion density screening diagnostic",
        "Bone scan scintigraphy technetium hot spot cold spot skeletal nuclear medicine",
        "Fluoroscopy barium contrast swallow esophagram small bowel follow through dynamic imaging",
        "Interventional radiology angiogram embolization percutaneous image guided biopsy drainage catheter",
        "DEXA dual energy absorptiometry bone mineral density T score Z score",
        "Doppler flow velocity waveform resistive index spectral analysis vascular imaging",
        "Radiologist report findings impression recommendation imaging modality protocol",
        "MRI spine disc herniation foraminal stenosis nerve root compression signal change vertebral",
        "Nuclear medicine thyroid scan iodine uptake scintigraphy radioiodine",
        "Radiograph imaging study contrast enhanced scout view field of view matrix acquisition",
        "CT pulmonary angiography CTPA filling defect pulmonary artery imaging protocol",
        "Radiology report reviewed images demonstrate no acute abnormality incidental finding noted",
        "MRI brain without with contrast sequences reviewed radiologist interpretation report",
    ],
    'Surgery': [
        "Laparoscopic cholecystectomy trocar port gallbladder dissection clipped divided specimen retrieved",
        "Open appendectomy incision peritoneum appendix ligated removed histopathology wound closure",
        "Hernia repair mesh inguinal femoral laparoscopic TEP TAPP suture fixation recurrence",
        "Thyroidectomy total partial parathyroid recurrent laryngeal nerve hemostasis drain placement",
        "Mastectomy breast cancer sentinel lymph node biopsy axillary dissection reconstruction flap",
        "Colectomy bowel resection anastomosis stoma colostomy ileostomy diverticulitis obstruction",
        "Knee replacement arthroplasty prosthesis cemented tibial femoral patella component",
        "Coronary artery bypass graft CABG saphenous vein internal mammary artery sternotomy",
        "Craniotomy brain tumor resection dura mater burr hole surgical excision biopsy",
        "Exploratory laparotomy peritonitis bowel obstruction adhesiolysis evisceration bleeding",
        "Splenectomy splenic rupture trauma portal hypertension platelet count spleen removal",
        "Nephrectomy kidney removal renal cell carcinoma laparoscopic radical partial",
        "Gastrectomy stomach resection ulcer bleeding partial total Billroth anastomosis",
        "Wound debridement necrotic tissue irrigation suture closure skin graft flap coverage",
        "Postoperative care wound healing drain output vital signs pain management surgical recovery",
        "Laparoscopic cholecystectomy performed trocar port inserted gallbladder dissected clipped divided specimen retrieved wound closed sutures postoperative recovery drain",
        "Operative note patient taken to operating room general anesthesia induced prepped draped sterile incision made fascia divided organ identified excised hemostasis closure",
        "Surgical resection performed specimen sent pathology intraoperative bleeding controlled suture ligation electrocautery drain placed wound irrigated closed subcuticular",
    ],
    'General Medicine': [
        "Diabetes mellitus type 2 HbA1c fasting glucose metformin insulin glycemic control diet exercise",
        "Hypertension diabetes dyslipidemia metabolic syndrome cardiovascular risk statin antihypertensive",
        "Fever malaise fatigue body ache viral infection influenza paracetamol antipyretic supportive care",
        "Urinary tract infection dysuria frequency urgency urine culture antibiotic trimethoprim nitrofurantoin",
        "Anemia hemoglobin low iron deficiency ferritin serum iron transferrin saturation iron supplementation",
        "Thyroid hypothyroidism TSH elevated thyroxine levothyroxine replacement therapy",
        "Rheumatoid arthritis joint pain swelling morning stiffness anti CCP rheumatoid factor methotrexate",
        "Osteoporosis bone density DEXA scan bisphosphonate alendronate calcium vitamin D fracture risk",
        "Obesity BMI weight management lifestyle modification bariatric surgery comorbidities",
        "Depression anxiety psychiatric evaluation antidepressant sertraline cognitive behavioral therapy",
        "Gout hyperuricemia uric acid joint pain allopurinol colchicine anti-inflammatory NSAID",
        "Vitamin B12 deficiency peripheral neuropathy megaloblastic anemia cyanocobalamin injection",
        "Sepsis infection bacteremia blood culture antibiotic IV fluids vasopressor ICU management",
        "Chronic kidney disease creatinine eGFR proteinuria dialysis nephrology referral",
        "General health checkup blood pressure glucose cholesterol CBC liver function kidney function routine",
    ],
}

# Build synthetic dataframe
syn_rows = []
for label, texts in synthetic.items():
    for text in texts:
        syn_rows.append({'text': text, 'label': label})

df_syn = pd.DataFrame(syn_rows)

# ── Combine mtsamples + synthetic ─────────────────────────────
df_combined = pd.concat([df_mt, df_syn], ignore_index=True)
print("\nCombined before balancing:")
print(df_combined['label'].value_counts())

# ── Balance to 1000 per class ──────────────────────────────────
balanced = []
for cat in categories:
    cat_df = df_combined[df_combined['label'] == cat].copy()
    if len(cat_df) == 0:
        print(f"WARNING: No samples for {cat}!")
        continue
    if len(cat_df) < 1000:
        cat_df = cat_df.sample(1000, replace=True, random_state=42)
    else:
        cat_df = cat_df.sample(1000, replace=False, random_state=42)
    balanced.append(cat_df)

df_balanced = pd.concat(balanced).sample(frac=1, random_state=42)
print("\nBalanced dataset:")
print(df_balanced['label'].value_counts())

# ── Split ─────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df_balanced, test_size=0.2, random_state=42, stratify=df_balanced['label']
)

# ── Pipeline ──────────────────────────────────────────────────
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=150000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        stop_words='english',
        analyzer='word'
    )),
    ('svm', LinearSVC(
        class_weight='balanced',
        max_iter=10000,
        C=5.0
    ))
])

# ── Train ─────────────────────────────────────────────────────
print("\nTraining model...")
pipeline.fit(train_df['text'], train_df['label'])
print("Training complete!")

# ── Evaluate ──────────────────────────────────────────────────
preds = pipeline.predict(test_df['text'])
print("\n--- Classification Report ---")
print(classification_report(test_df['label'], preds, zero_division=0))
correct = sum(p == t for p, t in zip(preds, test_df['label']))
print(f"Overall Accuracy: {correct}/{len(test_df)} = {correct/len(test_df)*100:.1f}%")

# ── Save ──────────────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
joblib.dump(pipeline, 'model/svm_pipeline.pkl')
joblib.dump(categories, 'model/labels.pkl')
print("\nModel saved successfully!")

# ── Quick test ────────────────────────────────────────────────
print("\n--- Quick Test ---")
samples = [
    ("Neurology",                   "Patient presents with seizures epilepsy EEG shows abnormal brain wave activity neurologist Parkinson tremor levodopa"),
    ("Gastroenterology",            "Colonoscopy performed colon polyp removed gastric ulcer Helicobacter pylori endoscopy abdominal pain diarrhea"),
    ("Cardiovascular / Pulmonary",  "ECG ST elevation myocardial infarction troponin elevated cardiac catheterization coronary artery stent heart failure"),
    ("Radiology",                   "CT scan chest X-ray MRI brain contrast bilateral pulmonary infiltrates pleural effusion radiograph imaging"),
    ("Surgery",                     "Laparoscopic cholecystectomy trocar port inserted gallbladder dissected clipped divided specimen retrieved wound closed sutures postoperative drain"),
    ("General Medicine",            "Diabetes mellitus type 2 HbA1c elevated metformin insulin blood pressure hypertension statin dyslipidemia"),
    ("Neurology",                   "Multiple sclerosis demyelination white matter lesions optic neuritis MRI brain neurologist interferon"),
    ("Gastroenterology",            "Liver cirrhosis hepatitis jaundice ascites hepatic encephalopathy lactulose spironolactone"),
    ("Cardiovascular / Pulmonary",  "Atrial fibrillation irregular rhythm anticoagulation warfarin rate control cardioversion ablation"),
    ("General Medicine",            "Fever malaise fatigue viral infection urinary tract infection anemia thyroid hypothyroidism TSH"),
]

print(f"{'Expected':<35} {'Predicted':<35} {'Result'}")
print("-" * 80)
correct_count = 0
for expected, text in samples:
    pred = pipeline.predict([text])[0]
    match = "✓ CORRECT" if pred == expected else "✗ WRONG"
    if pred == expected:
        correct_count += 1
    print(f"{expected:<35} {pred:<35} {match}")

print(f"\nQuick Test Score: {correct_count}/{len(samples)} = {correct_count/len(samples)*100:.0f}%")