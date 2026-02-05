import os
import numpy as np
from postprocess.calibrator import CECalibrator

#Create postprocess storage directory
postprocess_output_dir = "postprocess_store"
os.makedirs(postprocess_output_dir, exist_ok=True)

calibrator_path = os.path.join(postprocess_output_dir, "ce_calibrator.joblib")

#Load validation ground truth and CNN predictions
y_val_true = np.load("./data/validation/y_val_true.npy")
y_val_pred = np.load("./data/validation/y_val_pred_cnn.npy")

#Initialize calibrator with CE range
min_ce = float(y_val_true.min())
max_ce = float(y_val_true.max())

calibrator = CECalibrator(min_ce=min_ce, max_ce=max_ce)

#Fit calibrator
calibrator.fit(y_val_true, y_val_pred)

#Evaluate calibration
metrics = calibrator.evaluate(y_val_true, y_val_pred)

print("\nCalibration Metrics")
print(f"Brier before : {metrics['brier_before']:.6f}")
print(f"Brier after  : {metrics['brier_after']:.6f}")
print(f"Improvement  : {metrics['improvement']:.6f}")

#Save calibrator
calibrator.save(calibrator_path)

print(f"\nCalibrator saved at: {calibrator_path}")