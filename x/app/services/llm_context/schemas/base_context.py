from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from jinja2 import Template


class BaseContext(BaseModel, ABC):
    """Temel context sınıfı - tüm context'ler için ortak özellikler"""
    
    task_definition: str = Field(..., description="LLM'den ne istendiğini tanımlar")
    knowledge_context: Optional[str] = Field(None, description="RAG ile elde edilen bilgi parçacıkları")
    user_context: Optional[str] = Field(None, description="Kullanıcıya özgü bilgiler")
    output_format_context: Optional[str] = Field(None, description="İstenen çıktı formatı")
    
    @abstractmethod
    def to_prompt(self) -> str:
        """Context'i prompt metnine dönüştür"""
        pass
    
    @abstractmethod
    def to_system_prompt(self) -> str:
        """Context'i sistem prompt'una dönüştür"""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Context'i dictionary'e dönüştür"""
        return self.model_dump()
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Template değişkenlerini döndür"""
        return self.to_dict()
    
    def render_template(self, template_content: str) -> str:
        """Jinja2 template'ini context verileriyle doldur"""
        template = Template(template_content)
        return template.render(**self.get_template_variables())
