"""WhatsApp/SMS notification service for agricultural alerts."""

import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class NotificationConfig(BaseModel):
    """Configuration for notification channels."""

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""
    aws_sns_region: str = "ap-south-1"
    aws_sns_topic_arn: str = ""
    exotel_api_key: str = ""
    default_language: str = "en"


class AlertMessage(BaseModel):
    """Alert message structure."""

    alert_id: str
    plot_id: str
    farmer_id: str
    alert_type: str
    severity: str
    message_en: str
    message_local: Optional[str] = None
    language: str = "en"
    channel: str = "whatsapp"  # whatsapp, sms, voice_call
    metadata: dict[str, Any] = Field(default_factory=dict)


class NotificationResult(BaseModel):
    """Result of notification delivery."""

    success: bool
    message_id: Optional[str] = None
    channel: str
    error: Optional[str] = None


class IndicNLPTranslator:
    """Simple translator for Indic languages (placeholder for real NLP)."""

    def __init__(self):
        self.translations = {
            "hi": {
                "irrigation_needed": "सिंचाई की आवश्यकता है",
                "pest_detected": "कीट का पता चला",
                "weather_alert": "मौसम चेतावनी",
            },
            "bn": {
                "irrigation_needed": "সেচের প্রয়োজন",
                "pest_detected": "পোকা শনাক্ত করা হয়েছে",
                "weather_alert": "আবহাওয়া সতর্কতা",
            },
        }

    def translate(self, message: str, target_language: str) -> str:
        """Translate message to target language."""
        if target_language == "en":
            return message
        
        # Simple keyword-based translation (placeholder)
        if "irrigation" in message.lower():
            return self.translations.get(target_language, {}).get(
                "irrigation_needed", message
            )
        elif "pest" in message.lower():
            return self.translations.get(target_language, {}).get(
                "pest_detected", message
            )
        elif "weather" in message.lower():
            return self.translations.get(target_language, {}).get(
                "weather_alert", message
            )
        
        return message  # Return original if no translation found


class TwilioNotifier:
    """Twilio integration for WhatsApp and SMS notifications."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.client = None  # Would initialize twilio.rest.Client
        logger.info("Twilio notifier initialized")

    def send_whatsapp(self, to_number: str, message: str) -> NotificationResult:
        """Send WhatsApp message via Twilio."""
        try:
            # Placeholder for actual Twilio API call
            # client.messages.create(
            #     from_=f"whatsapp:{self.config.twilio_whatsapp_number}",
            #     to=f"whatsapp:{to_number}",
            #     body=message,
            # )
            logger.info(f"WhatsApp sent to {to_number}: {message[:50]}...")
            return NotificationResult(
                success=True,
                message_id=f"wa_{hash(message)}",
                channel="whatsapp",
            )
        except Exception as e:
            logger.error(f"WhatsApp send failed: {e}")
            return NotificationResult(
                success=False,
                channel="whatsapp",
                error=str(e),
            )

    def send_sms(self, to_number: str, message: str) -> NotificationResult:
        """Send SMS via Twilio."""
        try:
            # Placeholder for actual Twilio API call
            # client.messages.create(
            #     from_=self.config.twilio_number,
            #     to=to_number,
            #     body=message,
            # )
            logger.info(f"SMS sent to {to_number}: {message[:50]}...")
            return NotificationResult(
                success=True,
                message_id=f"sms_{hash(message)}",
                channel="sms",
            )
        except Exception as e:
            logger.error(f"SMS send failed: {e}")
            return NotificationResult(
                success=False,
                channel="sms",
                error=str(e),
            )


class AWSSNSNotifier:
    """AWS SNS integration for scalable notifications."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.client = None  # Would initialize boto3 SNS client
        logger.info("AWS SNS notifier initialized")

    def publish_to_topic(self, message: str, subject: str) -> NotificationResult:
        """Publish message to SNS topic."""
        try:
            # Placeholder for actual SNS API call
            # client.publish(
            #     TopicArn=self.config.aws_sns_topic_arn,
            #     Message=message,
            #     Subject=subject,
            # )
            logger.info(f"SNS published: {subject} - {message[:50]}...")
            return NotificationResult(
                success=True,
                message_id=f"sns_{hash(message)}",
                channel="sns",
            )
        except Exception as e:
            logger.error(f"SNS publish failed: {e}")
            return NotificationResult(
                success=False,
                channel="sns",
                error=str(e),
            )


class NotificationService:
    """Main notification orchestration service."""

    def __init__(self, config: Optional[NotificationConfig] = None):
        self.config = config or NotificationConfig()
        self.twilio = TwilioNotifier(self.config)
        self.sns = AWSSNSNotifier(self.config)
        self.translator = IndicNLPTranslator()
        self.dlq = []  # Dead letter queue for failed notifications

    def get_farmer_contact(self, farmer_id: str) -> dict[str, str]:
        """Get farmer contact information (placeholder)."""
        # Would query database
        return {
            "farmer_id": farmer_id,
            "phone": "+919876543210",
            "preferred_language": "hi",
            "preferred_channel": "whatsapp",
        }

    def format_alert_message(self, alert: AlertMessage) -> str:
        """Format alert into user-friendly message."""
        severity_emoji = {
            "low": "ℹ️",
            "medium": "⚠️",
            "high": "🚨",
            "critical": "🆘",
        }.get(alert.severity, "⚠️")

        message = f"""{severity_emoji} *कृषि सतर्कता | Agricultural Alert*

📍 Plot: {alert.plot_id}
🔔 Type: {alert.alert_type.replace('_', ' ').title()}
⚡ Severity: {alert.severity.upper()}

📝 {alert.message_en}

💡 Recommendation: Take immediate action.
"""
        return message

    def send_alert(self, alert: AlertMessage) -> NotificationResult:
        """Send alert through appropriate channel."""
        # Get farmer contact info
        contact = self.get_farmer_contact(alert.farmer_id)
        
        # Translate message if needed
        preferred_language = contact.get("preferred_language", "en")
        if preferred_language != "en" and alert.message_local is None:
            alert.message_local = self.translator.translate(
                alert.message_en, preferred_language
            )
        
        # Format message
        message = self.format_alert_message(alert)
        if alert.message_local:
            message += f"\n\n{alert.message_local}"
        
        # Determine channel
        channel = alert.channel or contact.get("preferred_channel", "whatsapp")
        phone = contact.get("phone", "")
        
        # Send notification
        result = None
        if channel == "whatsapp":
            result = self.twilio.send_whatsapp(phone, message)
        elif channel == "sms":
            # SMS needs shorter message
            sms_message = f"Alert: {alert.alert_type}. {alert.message_en[:100]}"
            result = self.twilio.send_sms(phone, sms_message)
        elif channel == "voice_call":
            # Would integrate with Exotel for voice calls
            logger.info(f"Voice call initiated to {phone}")
            result = NotificationResult(
                success=True,
                message_id=f"voice_{hash(alert.alert_id)}",
                channel="voice_call",
            )
        else:
            # Fallback to SNS topic
            result = self.sns.publish_to_topic(
                message, subject=f"Agri Alert: {alert.alert_type}"
            )
        
        # Handle failures with retry logic
        if not result.success:
            logger.warning(f"Notification failed, adding to DLQ: {alert.alert_id}")
            self.dlq.append({
                "alert": alert.dict(),
                "error": result.error,
                "timestamp": "2024-01-15T10:00:00Z",
            })
        
        return result

    def send_bulk_alerts(self, alerts: list[AlertMessage]) -> list[NotificationResult]:
        """Send multiple alerts in batch."""
        results = []
        for alert in alerts:
            result = self.send_alert(alert)
            results.append(result)
        
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Bulk send: {success_count}/{len(alerts)} successful")
        return results

    def get_dlq_messages(self) -> list[dict[str, Any]]:
        """Get messages from dead letter queue."""
        return self.dlq

    def retry_dlq(self) -> int:
        """Retry sending failed messages from DLQ."""
        retried = 0
        messages_to_retry = self.dlq.copy()
        self.dlq.clear()
        
        for item in messages_to_retry:
            alert_data = item["alert"]
            alert = AlertMessage(**alert_data)
            result = self.send_alert(alert)
            if result.success:
                retried += 1
            else:
                # Put back in DLQ
                self.dlq.append(item)
        
        logger.info(f"DLQ retry: {retried} messages succeeded")
        return retried


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    config = NotificationConfig(
        twilio_account_sid="ACxxxxxxxx",
        twilio_auth_token="your_token",
        twilio_whatsapp_number="+14155238886",
    )
    
    service = NotificationService(config)
    
    alert = AlertMessage(
        alert_id="alert_001",
        plot_id="WB_123_45",
        farmer_id="FARMER_1234",
        alert_type="low_soil_moisture",
        severity="high",
        message_en="Soil moisture is critically low. Irrigation recommended within 24 hours.",
        language="hi",
        channel="whatsapp",
    )
    
    result = service.send_alert(alert)
    print(f"Notification result: {result}")
