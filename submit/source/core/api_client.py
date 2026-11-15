"""
KimHanNom API Client Module
Provides integration with the FIT HCMUS Sino-Nom OCR & Transliteration Cloud API.
Supports automated authentication, image upload, OCR line/bbox detection, and Sino-Nom to Quoc Ngu transliteration.
"""
import os
import sys
import requests
from typing import Dict, Any, Optional, List, Tuple
from core.config import KIMHANNOM_BASE_URL, KIMHANNOM_EMAIL, KIMHANNOM_PASSWORD


class KimHanNomClient:
    """Client for interacting with the KimHanNom API at FIT HCMUS.
    
    Handles:
      1. Session authentication via /account/login (using email & password)
      2. Image upload to obtain server file name (/api/web/clc-sinonom/image-upload)
      3. Deep learning OCR extraction of Sino-Nom text and bounding boxes (/api/web/clc-sinonom/image-ocr)
      4. Transliteration / phonetic transcription from Sino-Nom to Quoc Ngu (/api/web/clc-sinonom/sinonom-transliteration)
    """

    def __init__(
        self,
        base_url: str = KIMHANNOM_BASE_URL,
        email: str = KIMHANNOM_EMAIL,
        password: str = KIMHANNOM_PASSWORD,
        auto_login: bool = True
    ):
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.password = password
        self.token: Optional[str] = None

        self.session = requests.Session()
        # Strictly ensure User-Agent does NOT start with 'python-requests'
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
        )
        self.session.headers.update({"User-Agent": self.user_agent})

        if auto_login and self.email and self.password:
            self.login()

    def login(self, email: Optional[str] = None, password: Optional[str] = None) -> bool:
        """Authenticate with the server using credentials and extract the session token from cookies."""
        login_email = email or self.email
        login_pwd = password or self.password

        if not login_email or not login_pwd:
            raise ValueError("Credentials missing: email and password are required. Configure them in .env file.")

        login_url = f"{self.base_url}/account/login"
        payload = {
            "UserName": login_email,
            "Password": login_pwd
        }

        response = self.session.post(login_url, data=payload, allow_redirects=True)
        response.raise_for_status()

        token = self.session.cookies.get("token")
        if not token:
            raise RuntimeError("Authentication failed: 'token' cookie not received from server.")

        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {self.token}"})
        return True

    def _ensure_authenticated(self):
        """Ensure token exists or attempt re-authentication."""
        if not self.token:
            self.login()

    def upload_image(self, image_path: str) -> str:
        """Upload a local image file to obtain a server file name (Image ID).
        
        Args:
            image_path: Absolute or relative path to the image.
            
        Returns:
            server_file_name (str): Identifier stored on the server cache.
        """
        self._ensure_authenticated()

        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        upload_url = f"{self.base_url}/api/web/clc-sinonom/image-upload"
        filename = os.path.basename(image_path)

        with open(image_path, "rb") as f:
            files = {"image_file": (filename, f, "image/jpeg")}
            # Temporary headers without Content-Type to allow multipart boundary generation
            headers = {"User-Agent": self.user_agent, "Authorization": f"Bearer {self.token}"}
            response = requests.post(upload_url, headers=headers, files=files)

        response.raise_for_status()
        res_json = response.json()

        if res_json.get("code") != "000000" or not res_json.get("data"):
            err_msg = res_json.get("message")
            raise RuntimeError(f"Image upload failed with code {res_json.get('code')}: {err_msg}")

        server_file_name = res_json["data"]["file_name"]
        return server_file_name

    def ocr_image(
        self,
        server_file_name: str,
        ocr_id: int = 1,
        lang_type: int = 2,
        reading_direction: int = 1,
        epitaph: int = 0,
        tag: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform OCR recognition on a previously uploaded image.
        
        Args:
            server_file_name: The file_name received from upload_image.
            ocr_id: 1=vertical standard, 2=administrative, 3=scene/outdoor, 4=horizontal standard.
            lang_type: 0=unknown, 1=Han, 2=Nom.
            reading_direction: 1=vertical (default), 2=horizontal.
            epitaph: 0=standard document, 1=stele/epitaph.
            tag: Optional custom model_id.
            
        Returns:
            Dictionary with result_ocr_text, result_bbox, result_file_name.
        """
        self._ensure_authenticated()

        ocr_url = f"{self.base_url}/api/web/clc-sinonom/image-ocr"
        payload = {
            "file_name": server_file_name,
            "ocr_id": ocr_id,
            "lang_type": lang_type,
            "reading_direction": reading_direction,
            "epitaph": epitaph,
            "tag": tag
        }

        headers = {
            "User-Agent": self.user_agent,
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        response = requests.post(ocr_url, headers=headers, json=payload)
        response.raise_for_status()
        res_json = response.json()

        if res_json.get("code") != "000000" or not res_json.get("data"):
            err_msg = res_json.get("message")
            raise RuntimeError(f"OCR failed with code {res_json.get('code')}: {err_msg}")

        return res_json["data"]

    def transliterate(
        self,
        text: str,
        lang_type: int = 2,
        font_type: int = 1
    ) -> List[str]:
        """Convert Sino-Nom text into Vietnamese Quoc Ngu phonetic transcription.
        
        Args:
            text: Sino-Nom text line.
            lang_type: 1=Han, 2=Nom.
            font_type: 1=Standard.
            
        Returns:
            List of transliteration text segments.
        """
        self._ensure_authenticated()

        trans_url = f"{self.base_url}/api/web/clc-sinonom/sinonom-transliteration"
        payload = {
            "text": text,
            "lang_type": lang_type,
            "font_type": font_type
        }

        headers = {
            "User-Agent": self.user_agent,
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        response = requests.post(trans_url, headers=headers, json=payload)
        response.raise_for_status()
        res_json = response.json()

        if res_json.get("code") != "000000" or not res_json.get("data"):
            err_msg = res_json.get("message")
            raise RuntimeError(f"Transliteration failed with code {res_json.get('code')}: {err_msg}")

        return res_json["data"].get("result_text_transcription", [])

    def process_image_end_to_end(
        self,
        image_path: str,
        ocr_id: int = 1,
        lang_type: int = 2,
        reading_direction: int = 1
    ) -> Dict[str, Any]:
        """Execute complete end-to-end pipeline on a raw scan: Upload -> OCR -> Transliteration.
        
        Returns:
            Dictionary containing:
              - server_file_name: Image ID on server
              - lines: List of dicts, each with 'nom_text', 'quocngu_transcription', 'bbox'
              - total_lines: Count of detected lines
        """
        server_file_name = self.upload_image(image_path)
        ocr_data = self.ocr_image(
            server_file_name=server_file_name,
            ocr_id=ocr_id,
            lang_type=lang_type,
            reading_direction=reading_direction
        )

        raw_texts = ocr_data.get("result_ocr_text", [])
        raw_bboxes = ocr_data.get("result_bbox", [])

        lines = []
        for idx, text in enumerate(raw_texts):
            text_cleaned = text.strip()
            transcription = ""
            if text_cleaned:
                try:
                    trans_parts = self.transliterate(text_cleaned, lang_type=lang_type)
                    transcription = " ".join(trans_parts).strip()
                except Exception:
                    transcription = ""

            bbox = raw_bboxes[idx] if idx < len(raw_bboxes) else []
            lines.append({
                "line_index": idx + 1,
                "nom_text": text_cleaned,
                "quocngu_transcription": transcription,
                "bbox": bbox
            })

        return {
            "image_path": image_path,
            "server_file_name": server_file_name,
            "total_lines": len(lines),
            "lines": lines
        }

    def ocr_preprocessed_layer(
        self,
        img_bgr,
        book_title: str = "",
        target_script: str = "ch",
        is_vertical: bool = False
    ) -> Dict[str, Any]:
        """Preprocess image to isolate target script layer (removing grid lines and secondary ink)
        and invoke KimHanNom Cloud OCR API on the clean layer image.
        """
        import tempfile
        from core.preprocess import DocumentPreprocessor, cv2_imwrite_unicode

        clean_layer = DocumentPreprocessor.synthesize_preprocessed_layer_image(
            img_bgr, book_title=book_title, target_script=target_script
        )
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
                tmp_path = tf.name
            cv2_imwrite_unicode(tmp_path, clean_layer)
            srv_name = self.upload_image(tmp_path)
            ocr_id = 1 if is_vertical else 4
            read_dir = 1 if is_vertical else 2
            ocr_data = self.ocr_image(
                server_file_name=srv_name,
                ocr_id=ocr_id,
                lang_type=2,
                reading_direction=read_dir
            )
            return {
                "server_file_name": srv_name,
                "result_bbox": ocr_data.get("result_bbox", []),
                "result_ocr_text": ocr_data.get("result_ocr_text", [])
            }
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

