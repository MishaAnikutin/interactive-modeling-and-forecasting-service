from src.core.application.building_model.schemas.nhits import NhitsParams

base_nhits = NhitsParams(
    max_steps=30,
    early_stop_patience_steps=3,
    val_check_steps=50,
    learning_rate=1e-3,
    scaler_type="robust",
)