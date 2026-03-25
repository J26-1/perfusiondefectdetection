def log_epoch(epoch, train_loss, val_loss, val_dice):

    with open("training_log.csv", "a") as f:
        f.write(f"{epoch},{train_loss},{val_loss},{val_dice}\n")