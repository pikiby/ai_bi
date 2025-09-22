#!/usr/bin/env python3
"""
Анализатор связей Obsidian
Парсит файлы и извлекает связи (wikilinks, теги, вложения)
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json


class ObsidianLinksAnalyzer:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.links = {}  # файл -> список ссылок
        self.backlinks = {}  # файл -> список обратных ссылок
        self.tags = {}  # файл -> список тегов
        self.embeddings = {}  # файл -> список вложений
        
    def extract_links_from_content(self, content: str) -> Tuple[List[str], List[str], List[str]]:
        """Извлекает ссылки, теги и вложения из содержимого файла"""
        # Wikilinks: [[файл]], [[файл|текст]], [[файл#заголовок]]
        wikilink_pattern = r'\[\[([^\]\#\|]+)(?:\#[^\]\|]+)?(?:\|[^\]]+)?\]\]'
        wikilinks = re.findall(wikilink_pattern, content)
        
        # Теги: #тег, #вложенный/тег
        tag_pattern = r'#([a-zA-Zа-яА-Я0-9_\/]+)'
        tags = re.findall(tag_pattern, content)
        
        # Вложения: ![[файл]], ![[файл#заголовок]]
        embed_pattern = r'!\[\[([^\]\#\|]+)(?:\#[^\]\|]+)?(?:\|[^\]]+)?\]\]'
        embeddings = re.findall(embed_pattern, content)
        
        return wikilinks, tags, embeddings
    
    def analyze_file(self, file_path: Path) -> Dict:
        """Анализирует один файл"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            wikilinks, tags, embeddings = self.extract_links_from_content(content)
            
            return {
                'wikilinks': wikilinks,
                'tags': tags,
                'embeddings': embeddings,
                'file_size': len(content),
                'lines': len(content.split('\n'))
            }
        except Exception as e:
            print(f"Ошибка при анализе {file_path}: {e}")
            return {'wikilinks': [], 'tags': [], 'embeddings': [], 'file_size': 0, 'lines': 0}
    
    def analyze_vault(self) -> Dict:
        """Анализирует всю базу знаний"""
        results = {}
        
        # Находим все .md файлы
        md_files = list(self.vault_path.rglob("*.md"))
        
        print(f"Найдено {len(md_files)} .md файлов")
        
        for file_path in md_files:
            relative_path = file_path.relative_to(self.vault_path)
            results[str(relative_path)] = self.analyze_file(file_path)
        
        return results
    
    def build_link_graph(self, analysis_results: Dict) -> Dict:
        """Строит граф связей"""
        graph = {
            'nodes': [],
            'edges': [],
            'tags': set(),
            'orphaned_files': []
        }
        
        # Создаем узлы (файлы)
        for file_path, data in analysis_results.items():
            node = {
                'id': file_path,
                'name': Path(file_path).stem,
                'path': file_path,
                'wikilinks_count': len(data['wikilinks']),
                'tags_count': len(data['tags']),
                'embeddings_count': len(data['embeddings']),
                'size': data['file_size'],
                'lines': data['lines']
            }
            graph['nodes'].append(node)
            
            # Собираем теги
            graph['tags'].update(data['tags'])
        
        # Создаем рёбра (связи)
        for file_path, data in analysis_results.items():
            for link in data['wikilinks']:
                # Ищем файл, на который ссылаемся
                target_file = self.find_target_file(link, analysis_results)
                if target_file:
                    edge = {
                        'source': file_path,
                        'target': target_file,
                        'type': 'wikilink',
                        'label': link
                    }
                    graph['edges'].append(edge)
                else:
                    # Файл не найден - возможно, это внешняя ссылка
                    graph['orphaned_files'].append(link)
        
        graph['tags'] = list(graph['tags'])
        return graph
    
    def find_target_file(self, link: str, analysis_results: Dict) -> str:
        """Находит файл по ссылке"""
        # Пробуем разные варианты
        possible_names = [
            link,
            f"{link}.md",
            f"kb_{link}.md",
            f"d_{link}.md",
            f"t_{link}.md"
        ]
        
        for name in possible_names:
            for file_path in analysis_results.keys():
                if Path(file_path).stem == name.replace('.md', ''):
                    return file_path
        
        return None
    
    def generate_mermaid_diagram(self, graph: Dict) -> str:
        """Генерирует Mermaid диаграмму связей"""
        mermaid = ["graph TD"]
        
        # Добавляем узлы
        for node in graph['nodes']:
            node_id = node['id'].replace('/', '_').replace('.', '_').replace(' ', '_')
            label = f"{node['name']}<br/>({node['wikilinks_count']} ссылок)"
            mermaid.append(f'    {node_id}["{label}"]')
        
        # Добавляем рёбра
        for edge in graph['edges']:
            source_id = edge['source'].replace('/', '_').replace('.', '_').replace(' ', '_')
            target_id = edge['target'].replace('/', '_').replace('.', '_').replace(' ', '_')
            mermaid.append(f'    {source_id} --> {target_id}')
        
        return '\n'.join(mermaid)
    
    def generate_report(self, analysis_results: Dict, graph: Dict) -> str:
        """Генерирует текстовый отчёт"""
        report = []
        report.append("# Анализ связей Obsidian\n")
        
        # Общая статистика
        total_files = len(analysis_results)
        total_links = sum(len(data['wikilinks']) for data in analysis_results.values())
        total_tags = len(graph['tags'])
        
        report.append(f"## Общая статистика")
        report.append(f"- Файлов: {total_files}")
        report.append(f"- Всего ссылок: {total_links}")
        report.append(f"- Уникальных тегов: {total_tags}")
        report.append(f"- Связей в графе: {len(graph['edges'])}")
        report.append("")
        
        # Топ файлов по количеству ссылок
        report.append("## Топ файлов по количеству ссылок")
        sorted_files = sorted(
            analysis_results.items(),
            key=lambda x: len(x[1]['wikilinks']),
            reverse=True
        )
        
        for file_path, data in sorted_files[:10]:
            report.append(f"- `{file_path}`: {len(data['wikilinks'])} ссылок")
        report.append("")
        
        # Теги
        report.append("## Все теги")
        for tag in sorted(graph['tags']):
            report.append(f"- #{tag}")
        report.append("")
        
        # Сиротские ссылки
        if graph['orphaned_files']:
            report.append("## Неразрешённые ссылки")
            for link in sorted(set(graph['orphaned_files'])):
                report.append(f"- `{link}`")
            report.append("")
        
        return '\n'.join(report)


def main():
    """Пример использования"""
    analyzer = ObsidianLinksAnalyzer("/home/boris/Documents/Work/analytics/ai_bi/docs")
    
    print("Анализируем связи в Obsidian...")
    analysis_results = analyzer.analyze_vault()
    
    print("Строим граф связей...")
    graph = analyzer.build_link_graph(analysis_results)
    
    print("Генерируем отчёт...")
    report = analyzer.generate_report(analysis_results, graph)
    
    # Сохраняем отчёт
    with open("/home/boris/Documents/Work/analytics/ai_bi/test/obsidian_links_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Сохраняем Mermaid диаграмму
    mermaid_diagram = analyzer.generate_mermaid_diagram(graph)
    with open("/home/boris/Documents/Work/analytics/ai_bi/test/obsidian_links_diagram.md", 'w', encoding='utf-8') as f:
        f.write(f"# Диаграмма связей Obsidian\n\n```mermaid\n{mermaid_diagram}\n```\n")
    
    # Сохраняем JSON данные
    with open("/home/boris/Documents/Work/analytics/ai_bi/test/obsidian_links_data.json", 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_results': analysis_results,
            'graph': graph
        }, f, ensure_ascii=False, indent=2)
    
    print("Анализ завершён!")
    print(f"- Отчёт: obsidian_links_report.md")
    print(f"- Диаграмма: obsidian_links_diagram.md") 
    print(f"- Данные: obsidian_links_data.json")


if __name__ == "__main__":
    main()
