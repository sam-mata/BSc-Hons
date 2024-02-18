# Imports
import logging


# Get logger config level
def set_logger_config():
    user_input = input("📃 Use debug mode? Yes (y) | No (n) : ").lower()
    if user_input == "y":
        logging.getLogger().setLevel(logging.DEBUG)
        logging.debug("🔍 Debug mode enabled.")
    elif user_input == "n":
        logging.getLogger().setLevel(logging.WARNING)
    else:
        print("⛔ Invalid input, please try again.")
        set_logger_config()


# Run sub files
def run():
    user_input = input("Train (1), Test (2), Predict (3), or Exit (4)? : ")
    if user_input == "1":
        print("🚂 Train")
    elif user_input == "2":
        print("🔍 Test")
    elif user_input == "3":
        print("🔮 Predict")
    elif user_input == "4":
        logging.info("👋 Goodbye!")
    else:
        print("⛔ Invalid input, please try again.")
        run()


# Let script run as project entry point
if __name__ == "__main__":
    set_logger_config()
    run()
