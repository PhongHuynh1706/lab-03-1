from stable_baselines3.common.logger import configure

def setup_csv_logger(model, log_dir="./logs_vacuum/"):
    """
    Gắn logger CSV cho SB3 model
    """
    new_logger = configure(log_dir, ["csv"])
    model.set_logger(new_logger)
    return model