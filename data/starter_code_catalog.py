# the starter code for beginners 
STARTER_CODE_CATALOG = {
    "classification": """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions, average="weighted"))""",
    "regression": """from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("RMSE:", mean_squared_error(y_test, predictions, squared=False))
print("R2:", r2_score(y_test, predictions))""",
    "clustering": """from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features = df.drop(columns=["id"], errors="ignore")
scaled_features = StandardScaler().fit_transform(features)

model = KMeans(n_clusters=3, random_state=42, n_init="auto")
clusters = model.fit_predict(scaled_features)

print("Silhouette Score:", silhouette_score(scaled_features, clusters))""",
    "anomaly_detection": """from sklearn.ensemble import IsolationForest

features = df.drop(columns=["id"], errors="ignore")

model = IsolationForest(contamination="auto", random_state=42)
labels = model.fit_predict(features)

df["anomaly_label"] = labels
print(df["anomaly_label"].value_counts())""",
    "time_series_forecasting": """from sklearn.metrics import mean_absolute_error

series = df.sort_values("date")["target"]
window = 7
predictions = series.shift(1).rolling(window=window).mean()

valid = predictions.notna()
print("MAE:", mean_absolute_error(series[valid], predictions[valid]))""",
    "nlp": """from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X = df["text"]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("F1 Score:", f1_score(y_test, predictions, average="weighted"))""",
}
