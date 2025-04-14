# message_alerts.py
from twilio.rest import Client

# Twilio credentials (replace with your own)
TWILIO_ACCOUNT_SID = ""
TWILIO_AUTH_TOKEN = ""
TWILIO_PHONE_NUMBER = ""


class MessageAlerts:
    def send_challan(self, plate_number, phone_number, violation_data, fine, violation_type="Overloading"):
        """
        Send challan via Twilio SMS.

        Args:
            plate_number (str): License plate number.
            phone_number (str): Recipient's phone number.
            violation_data: Violation-specific data (e.g., number of riders, speed).
            fine (int): Fine amount.
            violation_type (str): Type of violation.

        Returns:
            tuple: (success, message) indicating if the message was sent and a status message.
        """
        # Construct message body based on violation type
        message_body = f"Traffic Violation Detected: {violation_type}\nLicense Plate: {plate_number}\n"

        if violation_type == "Overloading":
            riders = violation_data if violation_data is not None else 3
            message_body += f"Riders: {riders}\n"
        elif violation_type == "Overspeed":
            speed_over_limit = violation_data if violation_data is not None else 0
            message_body += f"Speed Over Limit: {speed_over_limit} km/h\n"

        message_body += f"Fine: ₹{fine}\nPlease pay within 7 days."

        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        try:
            message = client.messages.create(
                body=message_body,
                from_=TWILIO_PHONE_NUMBER,
                to=phone_number
            )
            return True, f"Challan sent to {phone_number} for plate {plate_number}"
        except Exception as e:
            return False, f"Failed to send challan: {str(e)}"
