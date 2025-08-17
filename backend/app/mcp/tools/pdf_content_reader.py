from typing import Dict, Any, List, Optional
from .base import BaseMCPTool
import os
import io
import base64
from PIL import Image
import pdfplumber
import PyPDF2
import magic


class PDFContentReaderTool(BaseMCPTool):
    """PDF içeriğini okuma ve ayrıştırma için MCP tool"""
    
    def get_name(self) -> str:
        return "read_pdf_content"
    
    def get_description(self) -> str:
        return "PDF içeriğini okur ve metin/görsel olarak ayrıştırır"
    
    def get_input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pdf_path": {
                    "type": "string",
                    "description": "PDF dosya yolu"
                },
                "page_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "integer", "minimum": 1},
                        "end": {"type": "integer", "minimum": 1}
                    },
                    "description": "Sayfa aralığı (opsiyonel)"
                },
                "extract_mode": {
                    "type": "string",
                    "enum": ["text_only", "images_only", "both"],
                    "default": "both",
                    "description": "Çıkarma modu"
                },
                "image_quality": {
                    "type": "string",
                    "enum": ["low", "medium", "high"],
                    "default": "medium",
                    "description": "Görsel kalitesi"
                },
                "preserve_layout": {
                    "type": "boolean",
                    "default": True,
                    "description": "Layout'u koru"
                }
            },
            "required": ["pdf_path"]
        }
    
    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """PDF içerik çıkarma mantığı"""
        
        pdf_path = arguments["pdf_path"]
        page_range = arguments.get("page_range")
        extract_mode = arguments.get("extract_mode", "both")
        image_quality = arguments.get("image_quality", "medium")
        preserve_layout = arguments.get("preserve_layout", True)
        
        # PDF dosyası kontrolü
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": "PDF dosyası bulunamadı",
                "pdf_path": pdf_path
            }
        
        # Dosya tipi kontrolü
        try:
            file_type = magic.from_file(pdf_path, mime=True)
            if file_type != "application/pdf":
                return {
                    "success": False,
                    "error": f"Geçersiz dosya tipi: {file_type}",
                    "expected": "application/pdf"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Dosya tipi kontrolü başarısız: {str(e)}"
            }
        
        try:
            # PDF bilgilerini al
            pdf_info = await self._get_pdf_info(pdf_path)
            
            # Sayfa aralığını belirle
            start_page = page_range.get("start", 1) if page_range else 1
            end_page = page_range.get("end", pdf_info["total_pages"]) if page_range else pdf_info["total_pages"]
            
            # İçerik çıkarma
            content = {
                "pdf_info": pdf_info,
                "extraction_settings": {
                    "extract_mode": extract_mode,
                    "page_range": {"start": start_page, "end": end_page},
                    "image_quality": image_quality,
                    "preserve_layout": preserve_layout
                },
                "pages": []
            }
            
            if extract_mode in ["text_only", "both"]:
                text_content = await self._extract_text_content(
                    pdf_path, start_page, end_page, preserve_layout
                )
                content["text_content"] = text_content
            
            if extract_mode in ["images_only", "both"]:
                image_content = await self._extract_images(
                    pdf_path, start_page, end_page, image_quality
                )
                content["images"] = image_content
            
            # Sayfa bazlı detaylı içerik
            page_details = await self._extract_page_details(
                pdf_path, start_page, end_page, extract_mode
            )
            content["pages"] = page_details
            
            return {
                "success": True,
                "content": content,
                "statistics": {
                    "total_pages_processed": end_page - start_page + 1,
                    "text_pages": len([p for p in page_details if p.get("text")]),
                    "image_count": len(content.get("images", [])),
                    "extraction_mode": extract_mode
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"PDF işleme hatası: {str(e)}",
                "pdf_path": pdf_path
            }
    
    async def _get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """PDF temel bilgilerini al"""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            info = {
                "total_pages": len(pdf_reader.pages),
                "title": pdf_reader.metadata.title if pdf_reader.metadata else None,
                "author": pdf_reader.metadata.author if pdf_reader.metadata else None,
                "creator": pdf_reader.metadata.creator if pdf_reader.metadata else None,
                "producer": pdf_reader.metadata.producer if pdf_reader.metadata else None,
                "creation_date": str(pdf_reader.metadata.creation_date) if pdf_reader.metadata and pdf_reader.metadata.creation_date else None,
                "is_encrypted": pdf_reader.is_encrypted,
                "file_size": os.path.getsize(pdf_path)
            }
            
            return info
    
    async def _extract_text_content(self, pdf_path: str, start_page: int, end_page: int, preserve_layout: bool) -> List[Dict[str, Any]]:
        """Metin içeriğini çıkar"""
        text_pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                
                if preserve_layout:
                    # Layout korunarak metin çıkar
                    text = page.extract_text(layout=True)
                    tables = page.extract_tables()
                else:
                    # Basit metin çıkarma
                    text = page.extract_text()
                    tables = []
                
                # Sayfa boyutları
                page_info = {
                    "page_number": page_num + 1,
                    "text": text or "",
                    "char_count": len(text) if text else 0,
                    "line_count": len(text.split('\n')) if text else 0,
                    "tables": tables,
                    "table_count": len(tables),
                    "page_dimensions": {
                        "width": float(page.width),
                        "height": float(page.height)
                    }
                }
                
                # Metin istatistikleri
                if text:
                    words = text.split()
                    page_info["word_count"] = len(words)
                    page_info["avg_word_length"] = sum(len(word) for word in words) / len(words) if words else 0
                else:
                    page_info["word_count"] = 0
                    page_info["avg_word_length"] = 0
                
                text_pages.append(page_info)
        
        return text_pages
    
    async def _extract_images(self, pdf_path: str, start_page: int, end_page: int, quality: str) -> List[Dict[str, Any]]:
        """Görselleri çıkar"""
        images = []
        
        # Kalite ayarları
        quality_settings = {
            "low": {"dpi": 72, "format": "JPEG", "quality": 60},
            "medium": {"dpi": 150, "format": "PNG", "quality": 85},
            "high": {"dpi": 300, "format": "PNG", "quality": 95}
        }
        
        settings = quality_settings.get(quality, quality_settings["medium"])
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                    page = pdf.pages[page_num]
                    
                    # Sayfa görselini al
                    try:
                        page_image = page.to_image(resolution=settings["dpi"])
                        
                        # PIL Image'e çevir
                        pil_image = page_image.original
                        
                        # Base64'e encode et
                        buffer = io.BytesIO()
                        pil_image.save(buffer, format=settings["format"], quality=settings.get("quality", 85))
                        image_base64 = base64.b64encode(buffer.getvalue()).decode()
                        
                        image_info = {
                            "page_number": page_num + 1,
                            "image_data": image_base64,
                            "format": settings["format"],
                            "dimensions": {
                                "width": pil_image.width,
                                "height": pil_image.height
                            },
                            "dpi": settings["dpi"],
                            "size_bytes": len(buffer.getvalue()),
                            "quality": quality
                        }
                        
                        images.append(image_info)
                        
                    except Exception as e:
                        print(f"Sayfa {page_num + 1} görsel çıkarma hatası: {e}")
                        continue
        
        except Exception as e:
            print(f"Görsel çıkarma genel hatası: {e}")
        
        return images
    
    async def _extract_page_details(self, pdf_path: str, start_page: int, end_page: int, extract_mode: str) -> List[Dict[str, Any]]:
        """Sayfa detaylarını çıkar"""
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in range(start_page - 1, min(end_page, len(pdf.pages))):
                page = pdf.pages[page_num]
                
                page_detail = {
                    "page_number": page_num + 1,
                    "dimensions": {
                        "width": float(page.width),
                        "height": float(page.height)
                    }
                }
                
                if extract_mode in ["text_only", "both"]:
                    # Metin analizi
                    text = page.extract_text()
                    page_detail["text"] = text or ""
                    page_detail["has_text"] = bool(text and text.strip())
                    
                    # Metin blokları
                    chars = page.chars
                    page_detail["char_count"] = len(chars)
                    
                    # Font analizi
                    if chars:
                        fonts = {}
                        for char in chars:
                            font_name = char.get('fontname', 'Unknown')
                            font_size = char.get('size', 0)
                            font_key = f"{font_name}_{font_size}"
                            fonts[font_key] = fonts.get(font_key, 0) + 1
                        
                        page_detail["fonts"] = fonts
                        page_detail["dominant_font"] = max(fonts.items(), key=lambda x: x[1])[0] if fonts else None
                    
                    # Tablolar
                    tables = page.extract_tables()
                    page_detail["tables"] = tables
                    page_detail["table_count"] = len(tables)
                
                # Görsel elementler
                images = page.images
                page_detail["embedded_images"] = len(images)
                
                # Çizgiler ve şekiller
                lines = page.lines
                curves = page.curves
                rects = page.rects
                
                page_detail["layout_elements"] = {
                    "lines": len(lines),
                    "curves": len(curves),
                    "rectangles": len(rects)
                }
                
                # Sayfa tipi tahmini
                page_detail["page_type"] = self._classify_page_type(page_detail)
                
                pages.append(page_detail)
        
        return pages
    
    def _classify_page_type(self, page_detail: Dict[str, Any]) -> str:
        """Sayfa tipini sınıflandır"""
        has_text = page_detail.get("has_text", False)
        table_count = page_detail.get("table_count", 0)
        image_count = page_detail.get("embedded_images", 0)
        char_count = page_detail.get("char_count", 0)
        
        if not has_text and image_count > 0:
            return "image_only"
        elif table_count > 0 and char_count > 100:
            return "table_heavy"
        elif char_count > 1000:
            return "text_heavy"
        elif has_text and image_count > 0:
            return "mixed_content"
        elif has_text:
            return "text_only"
        else:
            return "empty_or_unknown"