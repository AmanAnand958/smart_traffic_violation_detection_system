# fine_calculator.py
class FineCalculator:
    def __init__(self):
        # Define fine structures for different violation types
        self.fine_structure = {
            "Overloading": {"base_fine": 500, "extra_rider_penalty": 200},  # Base + per extra rider
            "Helmet Violation": {"base_fine": 1000},  # Flat fine
            "Illegal Parking": {"base_fine": 500},  # Flat fine
            "Overspeed": {"base_fine": 400, "speed_penalty": 50},  # Base + per km/h over limit
            "Phone Violation": {"base_fine": 1000},  # Flat fine
            "Signal Violation": {"base_fine": 500},  # Flat fine
            "Wrong Side Driving": {"base_fine": 1500}  # Flat fine
        }

    def calculate_fine(self, plate_number, violation_data, violation_type="Overloading"):
        """
        Calculate fine based on violation type and data.

        Args:
            plate_number (str): License plate number.
            violation_data: Violation-specific data (e.g., number of riders, speed, etc.).
            violation_type (str): Type of violation.

        Returns:
            int: Total fine amount.
        """
        if violation_type not in self.fine_structure:
            raise ValueError(f"Unknown violation type: {violation_type}")

        fine_info = self.fine_structure[violation_type]

        if violation_type == "Overloading":
            riders = violation_data if violation_data is not None else 3
            extra_riders = max(0, riders - 2)  # Overloading starts at 3 riders
            total_fine = fine_info["base_fine"] + (fine_info["extra_rider_penalty"] * extra_riders)
        elif violation_type == "Overspeed":
            speed_over_limit = violation_data if violation_data is not None else 0
            total_fine = fine_info["base_fine"] + (fine_info["speed_penalty"] * max(0, speed_over_limit))
        else:
            # For other violations (Helmet, Illegal Parking, Phone, Signal, Wrong Side), use a flat fine
            total_fine = fine_info["base_fine"]

        return total_fine