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

st.markdown(
	"""
	<style>
	.stApp {
		background: #f8fafc;
		color: #0f172a;
	}
	[data-testid="stSidebar"] {
		background: #f1f5f9;
	}
	[data-testid="stAppViewContainer"] {
		background: transparent;
	}
	.block-container {
		background: #ffffff;
		border: 1px solid #e2e8f0;
		border-radius: 18px;
		padding: 1.25rem 1.4rem;
		box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
	}
	.hero {
		background: #f8fafc;
		border-left: 4px solid #2563eb;
		border-radius: 12px;
		padding: 0.2rem 0.8rem 0.8rem 0.8rem;
		margin-bottom: 1rem;
	}
	.hero h1 {
		margin: 0;
		font-size: 2rem;
		color: #0f172a;
	}
	.hero p {
		margin-top: 0.35rem;
		margin-bottom: 0;
		font-size: 0.98rem;
		color: #475569;
	}
	.result-card {
		background: #f0fdf4;
		border: 1px solid #86efac;
		border-radius: 18px;
		padding: 1rem 1.1rem;
	}
	</style>
	""",
	unsafe_allow_html=True,
)

st.markdown(
	"""
	<div class="hero">
		<h1>Student Performance Predictor</h1>
		<p>Predict final marks using attendance, assignment, quiz, and assessment inputs.</p>
	</div>
	""",
	unsafe_allow_html=True,
)

try:
	df = load_data(DATA_PATH)
except FileNotFoundError as error:
	st.error(str(error))
	st.stop()

model, mae, r2 = load_or_train_model(df)

col1, col2, col3 = st.columns(3)
with col1:
	st.metric("Model", "XGBoost")
with col2:
	st.metric("Prediction Type", "Regression")
with col3:
	st.metric("Status", "Ready")

st.caption("This application uses a trained XGBoost regression model for student performance forecasting.")

if mae is not None and r2 is not None:
	st.success("Model is ready for prediction.")
else:
	st.info("Loaded pretrained model from models/student_model.json.")

st.sidebar.header("New Student Input")
st.sidebar.markdown("Enter the student’s academic signals to generate a prediction.")
attendance = st.sidebar.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=0.0)
assignment = st.sidebar.number_input("Assignment Score", min_value=0.0, max_value=100.0, value=0.0)
internal = st.sidebar.number_input("Internal Assessment Score", min_value=0.0, max_value=100.0, value=0.0)
quiz = st.sidebar.number_input("Quiz Score", min_value=0.0, max_value=100.0, value=0.0)
study_hours = st.sidebar.number_input("Study Hours Per Day", min_value=0.0, max_value=24.0, value=0.0)
performance_index = (attendance + assignment + internal + quiz) / 4

predict_button = st.sidebar.button("Predict Final Marks", type="primary")

if predict_button:
	new_student = pd.DataFrame(
		[
			[attendance, assignment, internal, quiz, study_hours, performance_index]
		],
		columns=FEATURE_COLUMNS,
	)
	prediction = model.predict(new_student)[0]
	prediction = float(max(0, min(100, prediction)))
	status = "Pass" if prediction >= 40 else "Fail"

	st.markdown(
		f"""
		<div class="result-card">
			<h2 style="margin: 0;">Prediction Result</h2>
			<p style="margin: 0.8rem 0 0.25rem; font-size: 1.1rem;"><strong>Predicted Final Marks:</strong> {prediction:.2f}</p>
			<p style="margin: 0;"><strong>Status:</strong> {status}</p>
		</div>
		""",
		unsafe_allow_html=True,
	)

st.markdown("---")
