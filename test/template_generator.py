#!/usr/bin/env python3
"""
Генератор файлов по шаблону
Берет шаблон (.tmpl) и данные (.md с YAML frontmatter) и создает новый файл
"""

import os
import yaml
import re
from pathlib import Path
from typing import Dict, Any, List


class TemplateGenerator:
    def __init__(self, templates_dir: str = None, data_dir: str = None, output_dir: str = None):
        self.templates_dir = Path(templates_dir) if templates_dir else Path("docs/_tempates")
        self.data_dir = Path(data_dir) if data_dir else Path("docs")
        self.output_dir = Path(output_dir) if output_dir else Path("test/output")
        
        # Создаем выходную папку если не существует
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_template(self, template_name: str) -> str:
        """Загружает шаблон из файла"""
        template_path = self.templates_dir / f"{template_name}.tmpl"
        if not template_path.exists():
            raise FileNotFoundError(f"Шаблон не найден: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def load_data_from_md(self, data_file: str) -> Dict[str, Any]:
        """Извлекает YAML frontmatter из .md файла"""
        data_path = self.data_dir / f"{data_file}.md"
        if not data_path.exists():
            raise FileNotFoundError(f"Файл с данными не найден: {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Парсим YAML frontmatter
        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                try:
                    return yaml.safe_load(yaml_content) or {}
                except yaml.YAMLError as e:
                    print(f"Ошибка парсинга YAML: {e}")
                    return {}
        
        return {}
    
    def process_handlebars_template(self, template: str, data: Dict[str, Any]) -> str:
        """Обрабатывает Handlebars-подобные шаблоны"""
        result = template
        
        # Обработка {{#each items}} ... {{/each}}
        each_pattern = r'\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}'
        
        def replace_each(match):
            var_name = match.group(1)
            template_content = match.group(2)
            
            if var_name in data and isinstance(data[var_name], list):
                items = []
                for item in data[var_name]:
                    item_result = template_content
                    # Заменяем {{this.field}} на значения из item
                    for key, value in item.items():
                        item_result = item_result.replace(f'{{{{this.{key}}}}}', str(value))
                    items.append(item_result)
                return '\n'.join(items)
            return ''
        
        result = re.sub(each_pattern, replace_each, result, flags=re.DOTALL)
        
        # Обработка простых переменных {{variable}}
        simple_pattern = r'\{\{(\w+)\}\}'
        
        def replace_simple(match):
            var_name = match.group(1)
            return str(data.get(var_name, f'{{{{{var_name}}}}}'))
        
        result = re.sub(simple_pattern, replace_simple, result)
        
        return result
    
    def generate_file(self, template_name: str, data_file: str, output_name: str = None) -> str:
        """Генерирует новый файл по шаблону и данным"""
        # Загружаем шаблон
        template = self.load_template(template_name)
        
        # Загружаем данные
        data = self.load_data_from_md(data_file)
        
        # Обрабатываем шаблон
        result = self.process_handlebars_template(template, data)
        
        # Определяем имя выходного файла
        if output_name is None:
            output_name = f"{template_name}_{data_file}"
        
        output_path = self.output_dir / f"{output_name}.md"
        
        # Сохраняем результат
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result)
        
        print(f"Файл создан: {output_path}")
        return str(output_path)
    
    def list_templates(self) -> List[str]:
        """Возвращает список доступных шаблонов"""
        if not self.templates_dir.exists():
            return []
        
        return [f.stem for f in self.templates_dir.glob("*.tmpl")]
    
    def list_data_files(self) -> List[str]:
        """Возвращает список доступных файлов с данными"""
        if not self.data_dir.exists():
            return []
        
        return [f.stem for f in self.data_dir.glob("*.md")]


def main():
    """Пример использования"""
    generator = TemplateGenerator()
    
    print("Доступные шаблоны:")
    for template in generator.list_templates():
        print(f"  - {template}")
    
    print("\nДоступные файлы с данными:")
    for data_file in generator.list_data_files():
        print(f"  - {data_file}")
    
    # Пример генерации
    try:
        # Генерируем файл используя шаблон nav_card_multi и данные из kb_mobile_report_rep_mobile_full
        output_file = generator.generate_file(
            template_name="nav_card_multi",
            data_file="kb_mobile_report_rep_mobile_full",
            output_name="generated_nav_card"
        )
        print(f"\nСгенерированный файл: {output_file}")
        
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == "__main__":
    main()
