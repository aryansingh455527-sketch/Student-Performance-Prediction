import os

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_performance_dataset (1).csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
MODEL_JSON_PATH = os.path.join(MODEL_DIR, "student_model.json")
FEATURE_COLUMNS = [
	"Attendance",
	"Assignment",
	"InternalAssessment",
	"Quiz",
	"StudyHours",
	"PerformanceIndex",
]


def load_data(path: str) -> pd.DataFrame:
	if not os.path.exists(path):
		raise FileNotFoundError(f"Dataset not found at {path}")

	df = pd.read_csv(path)
	df.drop_duplicates(inplace=True)
	df.fillna(df.mean(numeric_only=True), inplace=True)
	df["PerformanceIndex"] = (
		df["Attendance"]
		+ df["Assignment"]
		+ df["InternalAssessment"]
		+ df["Quiz"]
	) / 4
	return df


def build_and_save_model(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, float, float]:
	X = df[FEATURE_COLUMNS]
	y = df["FinalMarks"]

	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=42
	)

	model = xgb.XGBRegressor(random_state=42, n_estimators=100, verbosity=0)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	mae = mean_absolute_error(y_test, y_pred)
	r2 = r2_score(y_test, y_pred)

	os.makedirs(MODEL_DIR, exist_ok=True)
	try:
		joblib.dump(model, MODEL_PATH)
		model.save_model(MODEL_JSON_PATH)
	except Exception as exc:
		st.warning(f"Could not save the model files: {exc}")

	return model, mae, r2


def load_or_train_model(df: pd.DataFrame) -> tuple[xgb.XGBRegressor, float | None, float | None]:
	if os.path.exists(MODEL_JSON_PATH):
		try:
			model = xgb.XGBRegressor()
			model.load_model(MODEL_JSON_PATH)
			return model, None, None
		except Exception as exc:
			st.warning(f"Could not load {MODEL_JSON_PATH}: {exc}. Trying backup model.")

	if os.path.exists(MODEL_PATH):
		try:
			model = joblib.load(MODEL_PATH)
			return model, None, None
		except Exception as exc:
			st.warning(f"Could not load {MODEL_PATH}: {exc}. Training a new model.")

	return build_and_save_model(df)


st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("Student Performance Prediction App")

try:
	df = load_data(DATA_PATH)
except FileNotFoundError as error:
	st.error(str(error))
	st.stop()

st.subheader("Dataset Preview")
st.dataframe(df.head())

with st.expander("Data Summary"):
	st.write(df.describe())

model, mae, r2 = load_or_train_model(df)
if mae is not None and r2 is not None:
	st.success("Model trained and saved successfully.")
	st.write(f"MAE (test split): {mae:.2f}")
	st.write(f"R2 score (test split): {r2:.2f}")
else:
	st.info("Loaded pretrained model from models/student_model.json.")

st.sidebar.header("New Student Input")
attendance = st.sidebar.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0)
assignment = st.sidebar.number_input("Assignment Score", min_value=0.0, max_value=100.0, value=75.0)
internal = st.sidebar.number_input("Internal Assessment Score", min_value=0.0, max_value=100.0, value=75.0)
quiz = st.sidebar.number_input("Quiz Score", min_value=0.0, max_value=100.0, value=75.0)
study_hours = st.sidebar.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=2.0)
performance_index = (attendance + assignment + internal + quiz) / 4

if st.sidebar.button("Predict Final Marks"):
	new_student = pd.DataFrame(
		[
			[attendance, assignment, internal, quiz, study_hours, performance_index]
		],
		columns=FEATURE_COLUMNS,
	)
	prediction = model.predict(new_student)[0]
	prediction = float(max(0, min(100, prediction)))
	st.write("## Prediction Result")
	st.write(f"**Predicted Final Marks:** {prediction:.2f}")
	st.write("**Status:**", "Pass" if prediction >= 60 else "Fail")

st.markdown("---")
st.write(
	"This app uses an XGBoost regression model trained on the provided dataset. The model is saved as models/student_model.json and a joblib backup is kept in models/xgb_model.joblib."
)
