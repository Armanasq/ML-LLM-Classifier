سیستم طبقه‌بندی حساب‌های معاملاتی با استفاده از LLM

## 1. ایده اصلی

ایجاد یک سیستم طبقه‌بندی هوشمند برای حساب‌های معاملاتی است که از قدرت LLM در ترکیب با تکنیک‌های پردازش داده و یادگیری ماشین استفاده می‌کند. این سیستم به طور خاص برای طبقه‌بندی حساب‌های معاملاتی بر اساس عملکرد و ریسک آنها طراحی شده است.

### 1.1 اجزای اصلی سیستم

سیستم از چهار جزء اصلی تشکیل شده است:

1. **پردازش داده و feature engineering**: این بخش داده‌های خام را پردازش کرده و ویژگی‌های جدید و معنادار ایجاد می‌کند.
2. **تولید Embedding**: این بخش داده‌های پردازش شده را به بردارهای عددی با معنا تبدیل می‌کند.
3. **یافتن موارد مشابه**: این بخش برای هر نمونه جدید، نزدیک‌ترین نمونه‌های مشابه را در مجموعه داده آموزشی پیدا می‌کند.
4. **طبقه‌بندی با LLM**: این بخش از یک LLM برای تصمیم‌گیری نهایی در مورد طبقه هر نمونه استفاده می‌کند.

## 2. پردازش داده و feature engineering

### 2.1 خواندن و پردازش اولیه داده

داده‌ها از یک پایگاه داده SQLite خوانده می‌شوند:

```python
engine = create_engine("sqlite:///output_database.db")
data = pd.read_sql_table("data_table", engine)
data = data.head(103)  # استفاده از 103 نمونه اول
```

### 2.2 feature engineering

تابع `engineer_features` ویژگی‌های جدیدی را ایجاد می‌کند که روابط مهم بین متغیرهای موجود را نشان می‌دهند:

```python
def engineer_features(df):
    df_copy = df.copy()
    numeric_columns = ['profit', 'deposits', 'commission', 'equity', 'balance', 'win_ratio', 'dealing_num']
    for col in numeric_columns:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    df_copy['profit_to_deposit_ratio'] = df_copy.apply(lambda row: safe_divide(row['profit'], row['deposits']), axis=1)
    df_copy['commission_to_profit_ratio'] = df_copy.apply(lambda row: safe_divide(row['commission'], row['profit']), axis=1)
    df_copy['equity_to_balance_ratio'] = df_copy.apply(lambda row: safe_divide(row['equity'], row['balance']), axis=1)
    df_copy['win_loss_ratio'] = df_copy.apply(lambda row: safe_divide(row['win_ratio'], 1 - row['win_ratio']), axis=1)
    df_copy['avg_profit_per_trade'] = df_copy.apply(lambda row: safe_divide(row['profit'], row['dealing_num']), axis=1)
    
    return df_copy
```

این ویژگی‌های جدید شامل نسبت‌هایی مانند سود به سپرده، کمیسیون به سود، و میانگین سود هر معامله هستند.

## 3. تولید Embedding

برای تبدیل داده‌های پردازش شده به embedding، از مدل `BAAI/bge-m3` استفاده شده است:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    multi_process=False,
    model_kwargs={"device": 'cuda'},
    encode_kwargs={"normalize_embeddings": True}
)

def generate_embedding(data_row):
    return embeddings.embed_query(str(data_row))
```

## 4. یافتن موارد مشابه

برای یافتن نمونه‌های مشابه، از الگوریتم Nearest Neighbors استفاده شده است:

```python
nn = NearestNeighbors(n_neighbors=5, metric='cosine')
nn.fit(X_train_scaled)

def get_similar_cases(query_embedding, k=5):
    distances, indices = nn.kneighbors([query_embedding], n_neighbors=k)
    similar_cases = X_train_engineered.iloc[indices[0]]
    similar_classes = y_train.iloc[indices[0]]
    return similar_cases, similar_classes
```

این الگوریتم از فاصله کسینوسی برای یافتن 5 نمونه نزدیک به هر نمونه جدید استفاده می‌کند.

## 5. طبقه‌بندی با LLM

### 5.1 انتخاب و پیکربندی LLM

برای طبقه‌بندی نهایی، از مدل `llama3.1:8b-instruct-fp16` استفاده شده است:

```python
llm = OpenAILike(model='llama3.1:8b-instruct-fp16',
                 api_base='http://localhost:11434/v1',
                 api_key='ollama',
                 temperature=0)
```


### 5.2 طراحی پرامپت

پرامپت طراحی شده برای هدایت LLM شامل چندین بخش است:

1. **زمینه**: توضیح وظیفه طبقه‌بندی و معنای کلاس‌ها
2. **اطلاعات کلیدی**: شامل ویژگی‌های نمونه مورد نظر، موارد مشابه، توزیع کلاس‌ها و اهمیت ویژگی‌ها
3. **توضیحات ویژگی‌ها**: شرح مختصر هر ویژگی
4. **دستورالعمل‌ها**: مراحل تحلیل و تصمیم‌گیری
5. **درخواست خروجی**: درخواست برای ارائه فقط طبقه نهایی

### 5.3 فرآیند طبقه‌بندی

تابع `classify_with_llm` مسئول ایجاد پرامپت و استفاده از LLM برای طبقه‌بندی است:

```python
def classify_with_llm(query_case, similar_cases, similar_classes):
    # ایجاد اطلاعات مورد نیاز برای پرامپت
    class_distribution = similar_cases.value_counts().to_dict()
    most_common_class = similar_classes.mode().values[0]
    feature_importance = X_train_engineered.corrwith(y_train).abs().sort_values(ascending=False).head(10).to_dict()

    # ایجاد پرامپت
    prompt = f"""As an expert financial analyst, classify this trading account based on its performance metrics. Use the provided information to determine the most appropriate class.

    Context:
    - Classes represent different levels of trading performance or risk.
    - Features describe various aspects of trading behavior and outcomes.
    - Similar cases are provided to inform your decision.

    Key Information:
    1. Query case features:
       {query_case.to_dict()}
    2. Similar cases (5 most similar accounts):
       {similar_cases.to_string()}
    3. Classes of similar cases: {similar_classes.tolist()}
    4. Class distribution among similar cases: {class_distribution}
    5. Most common class among similar cases: {most_common_class}
    6. Top 10 most important features and their correlation with the class:
       {feature_importance}

    Feature Descriptions:
    [توضیحات ویژگی‌ها]

    Instructions:
    1. Analyze the query case in relation to the similar cases.
    2. Consider the class distribution and the most common class among similar cases.
    3. Pay special attention to the values of the most important features in the query case.
    4. Use the additional engineered features to gain more insights.
    5. Determine the most likely class for the query case based on this comprehensive analysis.

    Important: Provide ONLY the final classification as your response. Do not include any explanations or reasoning.

    Classification:"""

    # استفاده از LLM برای طبقه‌بندی
    response = llm.complete(prompt)
    return response.text.strip()
```

این تابع برای هر نمونه جدید، یک پرامپت خاص ایجاد می‌کند که شامل اطلاعات آن نمونه و موارد مشابه آن است.

## 6. فرآیند کلی طبقه‌بندی

فرآیند کلی طبقه‌بندی برای هر نمونه جدید به صورت زیر است:

1. اعمال feature engineering بر روی نمونه
2. تولید embedding برای نمونه
3. یافتن موارد مشابه در مجموعه داده آموزشی
4. ایجاد پرامپت با استفاده از اطلاعات نمونه و موارد مشابه
5. استفاده از LLM برای تعیین طبقه نهایی

این فرآیند در تابع `llm_classifier` پیاده‌سازی شده است:

```python
def llm_classifier(X, y):
    y_pred = []
    for _, row in X.iterrows():
        test_case = engineer_features(pd.DataFrame(row).transpose()).iloc[0]
        test_embedding = generate_embedding(test_case)
        test_embedding_scaled = scaler.transform([test_embedding])
        similar_cases, similar_classes = get_similar_cases(test_embedding_scaled[0])
        classification = classify_with_llm(test_case, similar_cases, similar_classes)
        y_pred.append(float(classification))
    return np.array(y_pred)
```

## 7. ارزیابی مدل

برای ارزیابی عملکرد مدل، از چندین معیار و تکنیک استفاده شده است:

1. **معیارهای ارزیابی**: دقت، صحت، فراخوانی، F1-score، ضریب کاپای کوهن و ضریب همبستگی متیوس
2. **Confusion Matrix**: برای نمایش تعداد طبقه‌بندی‌های صحیح و اشتباه
3. **اعتبارسنجی متقابل K-Fold**: برای ارزیابی پایداری مدل بر روی زیرمجموعه‌های مختلف داده

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

for train_index, val_index in kf.split(X):
    X_train_cv, X_val_cv = X.iloc[train_index], X.iloc[val_index]
    y_train_cv, y_val_cv = y.iloc[train_index], y.iloc[val_index]
    
    y_pred_cv = llm_classifier(X_val_cv, y_val_cv)
    cv_scores.append(accuracy_score(y_val_cv, y_pred_cv))
```

## 8. تحلیل اهمیت ویژگی‌ها

برای درک بهتر تأثیر هر ویژگی بر طبقه‌بندی، همبستگی هر ویژگی با متغیر هدف محاسبه و نمایش داده شده است:

```python
feature_importance = X_train_engineered.corrwith(y_train).abs().sort_values(ascending=False)
plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance')
plt.xlabel('Absolute Correlation with Target')
plt.tight_layout()
plt.savefig('feature_importance.png')
```


این تحلیل به ما نشان می‌دهد که کدام ویژگی‌ها بیشترین تأثیر را بر طبقه‌بندی دارند. برای مثال، اگر ویژگی `order_duration_time` بالاترین همبستگی را با متغیر هدف داشته باشد، این نشان می‌دهد که مدت زمان سفارش‌ها یک عامل کلیدی در تعیین عملکرد حساب معاملاتی است.

ما همچنین این اطلاعات را در پرامپت LLM قرار دادیم تا به مدل کمک کنیم بر ویژگی‌های مهم‌تر تمرکز کند:

```python
feature_importance = X_train_engineered.corrwith(y_train).abs().sort_values(ascending=False).head(10).to_dict()
```

## 9. مقایسه توزیع داده‌ها در مجموعه‌های آموزش و آزمون

برای اطمینان از اینکه مجموعه‌های آموزش و آزمون نماینده مناسبی از کل داده‌ها هستند، ما توزیع ویژگی‌های مهم را در هر دو مجموعه مقایسه کردیم:

```python
def compare_distributions(train_data, test_data, feature):
    plt.figure(figsize=(10, 6))
    
    if pd.api.types.is_numeric_dtype(train_data[feature]):
        sns.histplot(train_data[feature], kde=True, label='Train', alpha=0.5)
        sns.histplot(test_data[feature], kde=True, label='Test', alpha=0.5)
    else:
        train_counts = train_data[feature].value_counts(normalize=True)
        test_counts = test_data[feature].value_counts(normalize=True)
        
        bar_width = 0.35
        index = np.arange(len(train_counts.index.union(test_counts.index)))
        
        plt.bar(index, train_counts.reindex(index).fillna(0), bar_width, label='Train', alpha=0.5)
        plt.bar(index + bar_width, test_counts.reindex(index).fillna(0), bar_width, label='Test', alpha=0.5)
        plt.xticks(index + bar_width / 2, index)
    
    plt.title(f'Distribution of {feature} in Train and Test Sets')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{feature}_distribution.png')
    plt.close()

for feature in X_train_engineered.columns[:5]:  # Top 5 features (including categorical)
    compare_distributions(X_train_engineered, X_test_engineered, feature)
```


## 10. پیش‌پردازش داده‌ها

قبل از استفاده از داده‌ها برای آموزش و آزمایش، چندین مرحله پیش‌پردازش انجام شد:

1. **تبدیل داده‌های غیرعددی**: برخی از ستون‌ها ممکن است حاوی داده‌های غیرعددی باشند. این مقادیر را به اعداد تبدیل کردیم:

```python
numeric_columns = ['profit', 'deposits', 'commission', 'equity', 'balance', 'win_ratio', 'dealing_num']
for col in numeric_columns:
    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
```

2. **مقیاس‌بندی داده‌ها**: برای اطمینان از اینکه تمام ویژگی‌ها در مقیاس یکسانی هستند، ما از `StandardScaler` استفاده کردیم:

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_embedded)
X_test_scaled = scaler.transform(X_test_embedded)
```

3. **مدیریت مقادیر گمشده**: در صورت وجود مقادیر گمشده، ما از روش `safe_divide` استفاده کردیم که مقادیر پیش‌فرض را برای تقسیم‌های نامعتبر برمی‌گرداند:

```python
def safe_divide(a, b, fill_value=0):
    try:
        return float(a) / (float(b) + 1e-5)
    except (ValueError, TypeError):
        return fill_value
```

## 11. استراتژی مدیریت داده‌های نامتوازن

اگرچه در این مجموعه داده خاص، عدم توازن شدیدی وجود نداشت (56 نمونه برای کلاس 1.0 و 47 نمونه برای کلاس 0.0)، همچنان از چند استراتژی برای اطمینان از عملکرد خوب مدل در هر دو کلاس استفاده کردیم:

1. **استفاده از معیارهای ارزیابی مناسب**: از معیارهایی مانند F1-score و Matthews Correlation Coefficient استفاده کردیم که نسبت به عدم توازن کلاس‌ها حساس هستند.

2. **ارائه اطلاعات توزیع کلاس به LLM**: در پرامپت، ما توزیع کلاس‌ها را برای نمونه‌های مشابه ارائه کردیم تا LLM بتواند این اطلاعات را در تصمیم‌گیری خود در نظر بگیرد.

## 12. مکانیسم تصمیم‌گیری LLM

مکانیسم تصمیم‌گیری LLM بر اساس اطلاعات ارائه شده در پرامپت است. LLM این اطلاعات را تحلیل می‌کند و با استفاده از دانش قبلی خود درباره تحلیل مالی، تصمیم می‌گیرد که نمونه مورد نظر به کدام کلاس تعلق دارد. 

اگرچه ما نمی‌توانیم دقیقاً ببینیم که LLM چگونه به تصمیم خود می‌رسد، اما با طراحی دقیق پرامپت، ما سعی کردیم LLM را هدایت کنیم تا:

1. ویژگی‌های نمونه مورد نظر را با نمونه‌های مشابه مقایسه کند.
2. به ویژگی‌های مهم‌تر (بر اساس همبستگی با متغیر هدف) توجه بیشتری کند.
3. توزیع کلاس‌ها در نمونه‌های مشابه را در نظر بگیرد.
4. از ویژگی‌های مهندسی شده برای درک بهتر عملکرد حساب استفاده کند.

## 13. مدیریت خطاها و استثناها

برای اطمینان از پایداری سیستم، چندین مکانیسم مدیریت خطا پیاده‌سازی شد:

1. **مدیریت تقسیم بر صفر**: با استفاده از تابع `safe_divide`، از خطاهای تقسیم بر صفر جلوگیری شد.

2. **مدیریت مقادیر غیرعددی**: با استفاده از `pd.to_numeric(errors='coerce')`, مقادیر غیرعددی به NaN تبدیل شدند.
