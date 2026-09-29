from ml.features.feature_generator import DEFAULT_OUTPUT_PATH, generate_features


rows = generate_features()
print(f"Generated {len(rows)} feature rows at {DEFAULT_OUTPUT_PATH}")
