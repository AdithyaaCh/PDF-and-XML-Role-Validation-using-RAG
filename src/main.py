from typing import List, Tuple
from src.utils import normalize_role, fuzzy_match
from config.config import FUZZY_MATCH_THRESHOLD

class RoleComparer:
    def __init__(self, fuzzy_threshold=FUZZY_MATCH_THRESHOLD):
        self.fuzzy_threshold = fuzzy_threshold

    def compare_roles(self, xml_roles: List[str], pdf_roles: List[str]) -> Tuple[bool, List[str]]:
        """
        Compares roles from XML and PDF and determines if any XML roles are missing in the PDF.
        Returns (is_incorrect, missing_roles_from_pdf).
        """
        normalized_xml_roles = {normalize_role(role) for role in xml_roles}
        normalized_pdf_roles = {normalize_role(role) for role in pdf_roles}

        # Determine which roles from XML are missing in the PDF
        missing_normalized_roles = normalized_xml_roles - normalized_pdf_roles

        # Attempt fuzzy match if exact match is missing
        still_missing_roles = []
        for role in xml_roles:
            norm_role = normalize_role(role)
            if norm_role in normalized_pdf_roles:
                continue
            matched = False
            for pdf_role in pdf_roles:
                if fuzzy_match(role, pdf_role, self.fuzzy_threshold):
                    matched = True
                    break
            if not matched:
                still_missing_roles.append(role)

        is_incorrect = bool(still_missing_roles)
        return is_incorrect, sorted(list(set(still_missing_roles)))

    def generate_report(self, is_incorrect: bool, missing_roles: List[str], xml_roles: List[str], pdf_roles: List[str]):
        print("\n--- Role Comparison Report ---")
        print(f"Total Unique Roles in XML: {len(set(xml_roles))}")
        print(f"Total Unique Roles found in PDF: {len(set(pdf_roles))}")

        if is_incorrect:
            print("\n--- MISSING ROLES (Defined in XML but not found in PDF) ---")
            for role in missing_roles:
                print(f"- {role}")
            print("\nCONCLUSION: The PDF is missing required roles as defined in the XML.")
        else:
            print("\n--- PDF ROLES ARE COMPLETE ---")
            print("All roles defined in XML were found or matched in the PDF.")

        print("\n-----------------------------\n")