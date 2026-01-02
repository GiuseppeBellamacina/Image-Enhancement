"""
Telegram Notifier for Training Progress

This module provides functions to send formatted Telegram messages
during the training process to keep track of progress remotely.
"""

import html
from typing import Dict, Tuple
import traceback


def send_telegram_message(
    bot_token: str, chat_id: str, message: str, parse_mode: str = "HTML"
) -> Tuple[bool, str]:
    """
    Send a message to a Telegram bot.

    Args:
        bot_token: Telegram bot token (from @BotFather)
        chat_id: Chat ID where to send the message
        message: Message text to send
        parse_mode: Message parsing mode ('HTML' or 'Markdown')

    Returns:
        Tuple of (success: bool, result_message: str)
    """
    try:
        import requests
    except ImportError:
        return False, "requests library not installed. Install with: pip install requests"

    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    # Escape HTML special characters to prevent parsing errors
    if parse_mode == "HTML":
        escaped_message = html.escape(message)
    else:
        escaped_message = message

    payload = {"chat_id": chat_id, "text": escaped_message, "parse_mode": parse_mode}

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return True, "Message sent successfully!"
    except Exception as e:
        return False, f"Error sending message: {str(e)}"


def format_epoch_message(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    best_epoch: int,
    best_val_loss: float,
) -> str:
    """
    Format a training progress message for Telegram.

    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_metrics: Dictionary with training metrics (loss, l1, ssim, etc.)
        val_metrics: Dictionary with validation metrics
        lr: Current learning rate
        best_epoch: Best epoch so far
        best_val_loss: Best validation loss so far

    Returns:
        Formatted message string
    """
    message = f"""🚀 TRAINING UPDATE 🚀

📊 Epoch: {epoch}/{total_epochs}
{'━' * 30}

📈 Training Metrics:
  • Loss: {train_metrics.get('loss', 0):.4f}
  • L1: {train_metrics.get('l1', 0):.4f}
  • SSIM: {train_metrics.get('ssim', 0):.4f}

📉 Validation Metrics:
  • Loss: {val_metrics.get('loss', 0):.4f}
  • L1: {val_metrics.get('l1', 0):.4f}
  • SSIM: {val_metrics.get('ssim', 0):.4f}

⚙️ Learning Rate: {lr:.2e}

🏆 Best So Far:
  • Epoch: {best_epoch}
  • Val Loss: {best_val_loss:.4f}
"""
    return message


def format_completion_message(
    final_epoch: int,
    total_epochs: int,
    best_epoch: int,
    best_val_loss: float,
    training_time: float,
    stopped_early: bool = False,
) -> str:
    """
    Format a training completion message for Telegram.

    Args:
        final_epoch: Final epoch reached
        total_epochs: Total number of epochs planned
        best_epoch: Best epoch achieved
        best_val_loss: Best validation loss achieved
        training_time: Total training time in seconds
        stopped_early: Whether training stopped early (early stopping)

    Returns:
        Formatted message string
    """
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)

    status_emoji = "⏸️" if stopped_early else "✅"
    status_text = "EARLY STOPPED" if stopped_early else "COMPLETED"

    message = f"""{status_emoji} TRAINING {status_text} {status_emoji}

📊 Summary:
  • Final Epoch: {final_epoch}/{total_epochs}
  • Best Epoch: {best_epoch}
  • Best Val Loss: {best_val_loss:.4f}

⏱️ Training Time:
  {hours}h {minutes}m {seconds}s

{'🎯 Training stopped early due to no improvement.' if stopped_early else '🎉 Training completed successfully!'}
"""
    return message


def format_error_message(
    epoch: int,
    total_epochs: int,
    error: Exception,
    show_traceback: bool = True,
) -> str:
    """
    Format a training error message for Telegram.

    Args:
        epoch: Epoch where the error occurred
        total_epochs: Total number of epochs planned
        error: Exception that was raised
        show_traceback: Whether to include full traceback

    Returns:
        Formatted message string
    """
    error_type = type(error).__name__
    error_msg = str(error)

    message = f"""❌ TRAINING ERROR ❌

📊 Context:
  • Epoch: {epoch}/{total_epochs}
  • Error Type: {error_type}

🔥 Error Message:
{error_msg}
"""

    if show_traceback:
        tb = traceback.format_exc()
        # Limit traceback length for Telegram (max message length is 4096)
        if len(tb) > 500:
            tb = tb[-500:]
        message += f"\n📋 Traceback (last 500 chars):\n{tb}"

    return message


def send_epoch_notification(
    bot_token: str,
    chat_id: str,
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    best_epoch: int,
    best_val_loss: float,
) -> bool:
    """
    Send an epoch progress notification to Telegram.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID where to send the message
        epoch: Current epoch number
        total_epochs: Total number of epochs
        train_metrics: Training metrics dictionary
        val_metrics: Validation metrics dictionary
        lr: Current learning rate
        best_epoch: Best epoch so far
        best_val_loss: Best validation loss so far

    Returns:
        True if message was sent successfully, False otherwise
    """
    if not bot_token or not chat_id:
        return False

    message = format_epoch_message(
        epoch=epoch,
        total_epochs=total_epochs,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        lr=lr,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )

    success, _ = send_telegram_message(bot_token, chat_id, message)
    return success


def send_completion_notification(
    bot_token: str,
    chat_id: str,
    final_epoch: int,
    total_epochs: int,
    best_epoch: int,
    best_val_loss: float,
    training_time: float,
    stopped_early: bool = False,
) -> bool:
    """
    Send a training completion notification to Telegram.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID where to send the message
        final_epoch: Final epoch reached
        total_epochs: Total number of epochs planned
        best_epoch: Best epoch achieved
        best_val_loss: Best validation loss achieved
        training_time: Total training time in seconds
        stopped_early: Whether training stopped early

    Returns:
        True if message was sent successfully, False otherwise
    """
    if not bot_token or not chat_id:
        return False

    message = format_completion_message(
        final_epoch=final_epoch,
        total_epochs=total_epochs,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        training_time=training_time,
        stopped_early=stopped_early,
    )

    success, _ = send_telegram_message(bot_token, chat_id, message)
    return success


def send_error_notification(
    bot_token: str,
    chat_id: str,
    epoch: int,
    total_epochs: int,
    error: Exception,
    show_traceback: bool = True,
) -> bool:
    """
    Send a training error notification to Telegram.

    Args:
        bot_token: Telegram bot token
        chat_id: Chat ID where to send the message
        epoch: Epoch where the error occurred
        total_epochs: Total number of epochs planned
        error: Exception that was raised
        show_traceback: Whether to include full traceback

    Returns:
        True if message was sent successfully, False otherwise
    """
    if not bot_token or not chat_id:
        return False

    message = format_error_message(
        epoch=epoch,
        total_epochs=total_epochs,
        error=error,
        show_traceback=show_traceback,
    )

    success, _ = send_telegram_message(bot_token, chat_id, message)
    return success
