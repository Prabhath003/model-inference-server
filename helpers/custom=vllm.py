import os
import re
import json
import time
import asyncio
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from collections import defaultdict, Counter
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import multiprocessing as mp
from functools import partial
import logging
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
import gc

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleQwenProcessor:
    """Simplified Qwen processor without complex threading"""
   
    def __init__(self, model_name: str = "Qwen/Qwen2.5-14B-Instruct", num_instances: int = 3):
        self.model_name = model_name
        self.num_instances = num_instances
        self.models = []
        self.tokenizers = []
        self.current_instance = 0
       
        logger.info(f"Initializing {num_instances} Qwen model instances...")
        self._initialize_models()
   
    def _initialize_models(self):
        """Initialize model instances on different GPUs"""
        for i in range(self.num_instances):
            device = f"cuda:{i % 2}"  # Distribute across 2 GPUs
           
            logger.info(f"Loading model instance {i+1} on {device}")
           
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
           
            # Load model with memory optimization
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                use_cache=True,
                # torch_dtype="auto"
            )
           
            self.models.append(model)
            self.tokenizers.append(tokenizer)
           
            logger.info(f"Model instance {i+1} loaded successfully")
   
    def generate_response(self, prompt: str, instance_id: int = None) -> str:
        """Generate response using specified or round-robin instance"""
        if instance_id is None:
            instance_id = self.current_instance
            self.current_instance = (self.current_instance + 1) % self.num_instances
       
        model = self.models[instance_id]
        tokenizer = self.tokenizers[instance_id]
       
        try:
            # Generate response
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
           
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=4096,
                    temperature=0.3,
                    top_p=0.8,
                    top_k=50,
                    num_beams=1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            if hasattr(model, 'past_key_values'):
                model.past_key_values = None
            torch.cuda.empty_cache()
            gc.collect()
            response = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
            return response
           
        except Exception as e:
            logger.error(f"Error in model {instance_id}: {str(e)}")
            return f"Error generating response: {str(e)}"
   
    def process_batch_parallel(self, prompts: List[str]) -> List[str]:
        """Process a batch of prompts in parallel using ThreadPoolExecutor"""
        results = [None] * len(prompts)
       
        def process_single(prompt_data):
            idx, prompt = prompt_data
            instance_id = idx % self.num_instances
            try:
                response = self.generate_response(prompt, instance_id)
                return idx, response
            except Exception as e:
                logger.error(f"Error processing prompt {idx}: {str(e)}")
                return idx, f"Error: {str(e)}"
       
        if len(prompts) <= 3:
            results = []
            for i, prompt in enumerate(prompts):
                instance_id = i % self.num_instances
                response = self.generate_response(prompt, instance_id)
                results.append(response)
            return results
       
        else:
            with ThreadPoolExecutor(max_workers=self.num_instances) as executor:
                # Submit all tasks
                future_to_idx = {
                    executor.submit(process_single, (i, prompt)): i
                    for i, prompt in enumerate(prompts)
                }
               
                # Collect results with timeout
                for future in as_completed(future_to_idx, timeout=300):  # 5 minute timeout
                    try:
                        idx, response = future.result(timeout=60)  # 1 minute per task
                        results[idx] = response
                    except Exception as e:
                        idx = future_to_idx[future]
                        logger.error(f"Task {idx} failed: {str(e)}")
                        results[idx] = f"Error: {str(e)}"
       
        return results
   
    def cleanup(self):
        """Cleanup models and free GPU memory"""
        logger.info("Cleaning up models...")
        for model in self.models:
            del model
        torch.cuda.empty_cache()
        gc.collect()

from vllm import LLM, SamplingParams

class VLLMProcessor:
    def __init__(self):
        logger.info("Initializing vLLM processor...")
        try:
            self.llm = LLM(
                model="Qwen/Qwen2.5-14B-Instruct",
                tensor_parallel_size=2,  # Use both GPUs
                max_model_len=4096,
                gpu_memory_utilization=0.85,
                enable_chunked_prefill=True,
                max_num_batched_tokens=8192,
                enforce_eager=True,
                trust_remote_code=True
            )
           
            self.sampling_params = SamplingParams(
                temperature=0.3,
                top_p=0.8,
                top_k=50,
                max_tokens=2048,
                repetition_penalty=1.05
            )
            logger.info("vLLM processor initialized successfully")
        except Exception as e:
            logger.error(f"vLLM initialization failed: {e}")
            raise
   
    def process_batch_vllm(self, prompts: List[str]) -> List[str]:
        """Process entire batch at once with vLLM"""
        try:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [output.outputs[0].text for output in outputs]
        except Exception as e:
            logger.error(f"vLLM batch processing failed: {e}")
            return [f"Error: {str(e)}" for _ in prompts]
   
    def cleanup(self):
        """Cleanup vLLM resources"""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()
        gc.collect()
   
class FixedFinancialReportGenerator:
    def __init__(self, input_folder: str, output_folder: str):
        """Initialize the fixed report generator"""
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
       
        # Create subdirectories
        self.chunks_folder = self.output_folder / "chunks"
        self.charts_folder = self.output_folder / "charts"
        self.html_assets_folder = self.output_folder / "assets"
       
        for folder in [self.chunks_folder, self.charts_folder, self.html_assets_folder]:
            folder.mkdir(exist_ok=True)
       
        # Initialize tokenizer for chunking
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-14B-Instruct")
       
        # Initialize simplified model processor
        # self.model_processor = SimpleQwenProcessor()
        self.vllm_processor = VLLMProcessor()
        # Data storage
        self.all_inferences = []
        self.financial_data = defaultdict(list)
       
        logger.info("Fixed Financial Report Generator initialized")
   
    def read_markdown_file(self, filepath: Path) -> str:
        """Read markdown file content"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
   
    def split_by_headers(self, content: str, filename: str) -> List[Dict]:
        """Split markdown content by headers and create chunks"""
        header_pattern = r'^(#{1,4})\s+(.+)$'
       
        chunks = []
        lines = content.split('\n')
        current_chunk = {
            'header': 'Document Start',
            'level': 0,
            'content': [],
            'source': filename
        }
       
        for line in lines:
            match = re.match(header_pattern, line, re.MULTILINE)
            if match:
                # Save previous chunk if it has content
                if current_chunk['content']:
                    current_chunk['content'] = '\n'.join(current_chunk['content'])
                    chunks.append(current_chunk)
               
                # Start new chunk
                level = len(match.group(1))
                header = match.group(2)
                current_chunk = {
                    'header': header,
                    'level': level,
                    'content': [line],
                    'source': filename
                }
            else:
                current_chunk['content'].append(line)
       
        # Don't forget the last chunk
        if current_chunk['content']:
            current_chunk['content'] = '\n'.join(current_chunk['content'])
            chunks.append(current_chunk)
       
        return chunks
   
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.tokenizer.encode(text))
   
    def create_financial_prompt(self, chunk: Dict, chunk_index: int) -> str:
        """Create optimized prompt for financial analysis"""
        prompt = f"""<|im_start|>system
You are an expert financial analyst. Analyze the following document section and extract key information in JSON format.

Extract:
1. Financial metrics (revenue, profit, expenses, assets, liabilities, ratios)
2. Key dates and periods
3. Important decisions or resolutions
4. Compliance information
5. Risk factors
6. Strategic initiatives
7. Performance indicators

Be specific and quantitative where possible.<|im_end|>

<|im_start|>user
Document: {chunk['source']}
Section: {chunk['header']}

Content:
{chunk['content'][:3000]}  # Limit content to avoid token overflow

Provide JSON analysis:
{{
    "summary": "brief summary",
    "financial_data": {{"metric": "value"}},
    "key_dates": ["date1", "date2"],
    "insights": ["insight1", "insight2"],
    "recommendations": ["rec1", "rec2"],
    "concerns": ["concern1", "concern2"],
    "strategic_implications": ["impl1", "impl2"],
    "quantitative_data": {{"metric": value}},
    "compliance_notes": ["note1", "note2"],
    "risk_assessment": {{"risk_type": "assessment"}}
}}

Respond only with the JSON, no additional text.<|im_end|>

<|im_start|>assistant"""
        return prompt
   
    def parse_response_to_dict(self, response: str, chunk: Dict, chunk_index: int) -> Dict:
        """Parse model response to structured dictionary"""
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
            else:
                # Fallback parsing
                result = {
                    "summary": response[:500] + "..." if len(response) > 500 else response,
                    "financial_data": {},
                    "key_dates": [],
                    "insights": [],
                    "recommendations": [],
                    "concerns": [],
                    "strategic_implications": [],
                    "quantitative_data": {},
                    "compliance_notes": [],
                    "risk_assessment": {}
                }
        except json.JSONDecodeError:
            # If JSON parsing fails, create structured response from text
            result = {
                "summary": response[:500] + "..." if len(response) > 500 else response,
                "financial_data": {},
                "key_dates": [],
                "insights": [],
                "recommendations": [],
                "concerns": [],
                "strategic_implications": [],
                "quantitative_data": {},
                "compliance_notes": [],
                "risk_assessment": {}
            }
       
        # Add metadata
        result['source_document'] = chunk['source']
        result['section_header'] = chunk['header']
        result['chunk_index'] = chunk_index
        result['timestamp'] = datetime.now().isoformat()
       
        return result
   
    def process_all_documents_fixed(self):
        """Process all documents with fixed parallel processing"""
        output_md = self.output_folder / "aggregated_analysis.md"
       
        # Initialize the output file
        with open(output_md, 'w', encoding='utf-8') as f:
            f.write("# Comprehensive Financial Documents Analysis Report\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
       
        # Collect all chunks from all documents
        all_chunks = []
        md_files = list(self.input_folder.glob("*.md"))
       
        logger.info(f"Found {len(md_files)} documents to process")
       
        for md_file in md_files:
            logger.info(f"Reading document: {md_file.name}")
            content = self.read_markdown_file(md_file)
            chunks = self.split_by_headers(content, md_file.name)
            all_chunks.extend(chunks)
       
        logger.info(f"Total chunks to process: {len(all_chunks)}")
       
        # Process chunks in smaller batches to avoid hanging
        batch_size = 12  # Smaller batch size (2 per model instance)
        total_batches = (len(all_chunks) + batch_size - 1) // batch_size
       
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(all_chunks))
            batch_chunks = all_chunks[start_idx:end_idx]
           
            logger.info(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_chunks)} chunks)")
           
            # Create prompts for this batch
            prompts = []
            for i, chunk in enumerate(batch_chunks):
                chunk_index = start_idx + i
                prompt = self.create_financial_prompt(chunk, chunk_index)
                prompts.append(prompt)
           
            # Process batch with timeout and error handling
            try:
                start_time = time.time()
                # responses = self.model_processor.process_batch_parallel(prompts)
                responses = self.vllm_processor.process_batch_vllm(prompts)

                batch_time = time.time() - start_time
                logger.info(f"Batch {batch_num + 1} completed in {batch_time:.2f} seconds")
               
                # Process responses
                for i, (chunk, response) in enumerate(zip(batch_chunks, responses)):
                    chunk_index = start_idx + i
                   
                    if response and not response.startswith("Error"):
                        result = self.parse_response_to_dict(response, chunk, chunk_index)
                        self.all_inferences.append(result)
                        self.save_inference_to_md(result, output_md)
                       
                        # Collect financial data
                        for metric, value in result.get('financial_data', {}).items():
                            self.financial_data[metric].append({
                                'value': value,
                                'source': result['source_document'],
                                'section': result['section_header']
                            })
                       
                        # Also collect quantitative data
                        for metric, value in result.get('quantitative_data', {}).items():
                            self.financial_data[f"quant_{metric}"].append({
                                'value': value,
                                'source': result['source_document'],
                                'section': result['section_header']
                            })
                    else:
                        logger.warning(f"Failed to process chunk {chunk_index}: {response}")
               
                logger.info(f"Completed batch {batch_num + 1}/{total_batches}")
               
                # Brief pause between batches
                time.sleep(2)
               
            except Exception as e:
                logger.error(f"Error processing batch {batch_num + 1}: {str(e)}")
                continue
       
        logger.info(f"Processed {len(self.all_inferences)} chunks from {len(md_files)} documents")
        return output_md
   
    def save_inference_to_md(self, inference: Dict, output_file: Path):
        """Append inference to markdown file"""
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n## Source: {inference['source_document']} - {inference['section_header']}\n")
            f.write(f"*Processed at: {inference['timestamp']}*\n\n")
           
            f.write("### Summary\n")
            f.write(f"{inference['summary']}\n\n")
           
            if inference.get('financial_data'):
                f.write("### Financial Data\n")
                for metric, value in inference['financial_data'].items():
                    f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
           
            if inference.get('quantitative_data'):
                f.write("### Quantitative Data\n")
                for metric, value in inference['quantitative_data'].items():
                    f.write(f"- **{metric}**: {value}\n")
                f.write("\n")
           
            for section_name in ['key_dates', 'insights', 'strategic_implications', 'recommendations', 'concerns', 'compliance_notes']:
                if inference.get(section_name):
                    f.write(f"### {section_name.replace('_', ' ').title()}\n")
                    for item in inference[section_name]:
                        f.write(f"- {item}\n")
                    f.write("\n")
           
            if inference.get('risk_assessment'):
                f.write("### Risk Assessment\n")
                for risk_type, assessment in inference['risk_assessment'].items():
                    f.write(f"- **{risk_type}**: {assessment}\n")
                f.write("\n")
           
            f.write("---\n")
   
    def create_visualizations(self):
        """Create comprehensive charts and graphs"""
        logger.info("Creating visualizations...")
        charts_created = []
       
        try:
            # 1. Financial Metrics Overview
            if self.financial_data:
                metric_counts = {metric: len(values) for metric, values in self.financial_data.items()}
                top_metrics = sorted(metric_counts.items(), key=lambda x: x[1], reverse=True)[:20]
               
                if top_metrics:
                    fig = go.Figure()
                    metrics = [m[0] for m in top_metrics]
                    counts = [m[1] for m in top_metrics]
                   
                    fig.add_trace(go.Bar(
                        x=metrics,
                        y=counts,
                        marker_color='steelblue',
                        text=counts,
                        textposition='auto',
                    ))
                   
                    fig.update_layout(
                        title="Top 20 Financial Metrics by Frequency",
                        xaxis_title="Metrics",
                        yaxis_title="Frequency",
                        template="plotly_white",
                        height=600,
                        xaxis_tickangle=-45
                    )
                   
                    chart_path = self.charts_folder / "top_financial_metrics.html"
                    fig.write_html(str(chart_path))
                    charts_created.append(chart_path)
           
            # 2. Document Analysis Distribution
            doc_counts = Counter(inf['source_document'] for inf in self.all_inferences)
           
            if doc_counts:
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(doc_counts.keys()),
                        values=list(doc_counts.values()),
                        hole=0.4,
                        textinfo='label+percent'
                    )
                ])
               
                fig.update_layout(
                    title="Document Analysis Distribution",
                    template="plotly_white",
                    height=600
                )
               
                chart_path = self.charts_folder / "document_distribution.html"
                fig.write_html(str(chart_path))
                charts_created.append(chart_path)
           
            # 3. Analysis Summary
            insights_count = sum(len(inf.get('insights', [])) for inf in self.all_inferences)
            concerns_count = sum(len(inf.get('concerns', [])) for inf in self.all_inferences)
            recommendations_count = sum(len(inf.get('recommendations', [])) for inf in self.all_inferences)
            strategic_impl_count = sum(len(inf.get('strategic_implications', [])) for inf in self.all_inferences)
           
            categories = ['Insights', 'Concerns', 'Recommendations', 'Strategic Implications']
            values = [insights_count, concerns_count, recommendations_count, strategic_impl_count]
            colors = ['#2E86AB', '#F18F01', '#A23B72', '#6A994E']
           
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker_color=colors,
                    text=values,
                    textposition='auto'
                )
            ])
           
            fig.update_layout(
                title="Analysis Summary by Category",
                xaxis_title="Category",
                yaxis_title="Count",
                template="plotly_white",
                height=500
            )
           
            chart_path = self.charts_folder / "analysis_summary.html"
            fig.write_html(str(chart_path))
            charts_created.append(chart_path)
           
            logger.info(f"Created {len(charts_created)} visualizations")
           
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
       
        return charts_created
   
    def chunk_aggregated_content(self, md_file_path: Path, max_tokens: int = 6000) -> List[str]:
        """Chunk the aggregated markdown content for final processing"""
        with open(md_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
       
        # Split by document sections
        sections = re.split(r'^## Source:', content, flags=re.MULTILINE)[1:]
       
        chunks = []
        current_chunk = []
        current_tokens = 0
       
        for section in sections:
            section_tokens = self.count_tokens(section)
           
            if current_tokens + section_tokens > max_tokens and current_chunk:
                chunks.append('## Source:' + '\n## Source:'.join(current_chunk))
                current_chunk = [section]
                current_tokens = section_tokens
            else:
                current_chunk.append(section)
                current_tokens += section_tokens
       
        if current_chunk:
            chunks.append('## Source:' + '\n## Source:'.join(current_chunk))
       
        logger.info(f"Created {len(chunks)} chunks for final processing")
        return chunks
   
    def generate_executive_summary(self, chunks: List[str]) -> str:
        """Generate executive summary from chunked content"""
        logger.info("Generating executive summary...")
       
        # Use only first few chunks for summary to avoid overwhelming the model
        summary_chunks = chunks[:5]
       
        prompts = []
        for i, chunk in enumerate(summary_chunks):
            prompt = f"""<|im_start|>system
You are a Chief Financial Officer creating an executive summary. Analyze the financial data and create a comprehensive summary for senior management.<|im_end|>

<|im_start|>user
Based on the following financial analysis data, create an executive summary suitable for C-level executives.

Focus on:
1. Key financial performance highlights
2. Critical strategic insights
3. Major risk factors
4. Essential recommendations

Content:
{chunk[:2000]}  # Limit to avoid token overflow

Provide a structured executive summary that is concise yet comprehensive.<|im_end|>

<|im_start|>assistant"""
            prompts.append(prompt)
       
        try:
            # Process summary chunks
            # responses = self.model_processor.process_batch_parallel(prompts)
            responses = self.vllm_processor.process_batch_vllm(prompts)            
            # Combine valid responses
            valid_summaries = [r for r in responses if r and not r.startswith("Error")]
           
            if valid_summaries:
                # Create final combined summary
                combined_prompt = f"""<|im_start|>system
You are synthesizing multiple executive summaries into a final comprehensive report for the board of directors.<|im_end|>

<|im_start|>user
Synthesize the following executive summaries into a single, comprehensive executive summary.

Structure it with:
1. **Executive Overview** (2-3 paragraphs)
2. **Financial Performance Highlights**
3. **Strategic Insights & Opportunities**
4. **Risk Assessment & Mitigation**
5. **Priority Recommendations**
6. **Conclusion & Forward Outlook**

Summaries to synthesize:
{' '.join(valid_summaries[:3])}

Create a professional, actionable executive summary.<|im_end|>

<|im_start|>assistant"""
               
                final_summary = self.vllm_processor.process_batch_vllm([combined_prompt])[0]
                return final_summary
           
        except Exception as e:
            logger.error(f"Error generating executive summary: {str(e)}")
       
        return "Executive summary could not be generated due to processing errors."
   
    def generate_insights_analysis(self) -> str:
        """Generate detailed insights analysis"""
        logger.info("Generating insights analysis...")
       
        # Collect all insights
        all_insights = []
        all_strategic_implications = []
       
        for inf in self.all_inferences:
            all_insights.extend(inf.get('insights', []))
            all_strategic_implications.extend(inf.get('strategic_implications', []))
       
        # Remove duplicates and get top insights
        unique_insights = list(set(all_insights))[:50]
        unique_implications = list(set(all_strategic_implications))[:50]
       
        prompt = f"""<|im_start|>system
You are a senior strategic analyst creating a comprehensive insights report for executive leadership.<|im_end|>

<|im_start|>user
Analyze and synthesize the following insights and strategic implications from a comprehensive financial document analysis.

Insights (top 25):
{json.dumps(unique_insights[:25], indent=2)}

Strategic Implications (top 25):
{json.dumps(unique_implications[:25], indent=2)}

Create a structured analysis with:
1. **Thematic Analysis** - Group insights by business themes
2. **Strategic Impact Assessment** - Evaluate business implications
3. **Competitive Positioning** - Market insights
4. **Operational Excellence** - Process insights
5. **Financial Performance** - Financial trends
6. **Priority Action Items** - Key actions

Format as a comprehensive business intelligence report.<|im_end|>

<|im_start|>assistant"""
       
        try:
            # insights_analysis = self.model_processor.generate_response(prompt)
            insights_analysis = self.vllm_processor.process_batch_vllm([prompt])[0]
            return insights_analysis
        except Exception as e:
            logger.error(f"Error generating insights analysis: {str(e)}")
            return "Insights analysis could not be generated."
   
    def generate_risk_assessment(self) -> str:
        """Generate comprehensive risk assessment"""
        logger.info("Generating risk assessment...")
       
        # Collect all risks and concerns
        all_concerns = []
        all_risk_assessments = {}
       
        for inf in self.all_inferences:
            all_concerns.extend(inf.get('concerns', []))
            risk_assessment = inf.get('risk_assessment', {})
            for risk_type, assessment in risk_assessment.items():
                if risk_type not in all_risk_assessments:
                    all_risk_assessments[risk_type] = []
                all_risk_assessments[risk_type].append(assessment)
       
        unique_concerns = list(set(all_concerns))[:50]
       
        prompt = f"""<|im_start|>system
You are a Chief Risk Officer preparing a comprehensive risk assessment report for the board of directors.<|im_end|>

<|im_start|>user
Based on the following risk data from comprehensive financial document analysis, create a detailed risk assessment report.

Identified Concerns (top 25):
{json.dumps(unique_concerns[:25], indent=2)}

Risk Assessment Categories:
{json.dumps(dict(list(all_risk_assessments.items())[:15]), indent=2)}

Create a structured risk assessment with:
1. **Executive Risk Summary**
2. **Risk Category Analysis** - Financial, Operational, Strategic, Compliance
3. **Risk Prioritization** - High/Medium/Low impact
4. **Mitigation Strategies** - Recommended actions
5. **Risk Monitoring** - KPIs and indicators
6. **Contingency Planning** - Response plans

Format as a professional risk management report.<|im_end|>

<|im_start|>assistant"""
       
        try:
            # risk_assessment = self.model_processor.generate_response(prompt)
            risk_assessment = self.vllm_processor.process_batch_vllm([prompt])[0]
            return risk_assessment
        except Exception as e:
            logger.error(f"Error generating risk assessment: {str(e)}")
            return "Risk assessment could not be generated."
   
    def create_html_report(self, md_file_path: Path, charts: List[Path]):
        """Create comprehensive HTML report"""
        logger.info("Creating HTML report...")
       
        # Chunk the aggregated content
        chunks = self.chunk_aggregated_content(md_file_path)
       
        # Generate comprehensive sections
        exec_summary = self.generate_executive_summary(chunks)
        insights_analysis = self.generate_insights_analysis()
        risk_assessment = self.generate_risk_assessment()
       
        # Collect data for display
        doc_counts = Counter(inf['source_document'] for inf in self.all_inferences)
       
        # Financial metrics summary
        financial_metrics_summary = []
        for metric, values in list(self.financial_data.items())[:50]:
            if values:
                latest = values[-1]
                financial_metrics_summary.append({
                    'metric': metric,
                    'value': latest['value'],
                    'source': latest['source'],
                    'section': latest['section']
                })
       
        def format_content_for_html(content: str) -> str:
            """Format content for HTML display"""
            if not content:
                return "<p>Content not available.</p>"
           
            # Split content into paragraphs and format
            paragraphs = content.split('\n\n')
            formatted_content = ""
           
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                   
                # Check if it's a heading (starts with #)
                if para.startswith('##'):
                    heading = para.replace('##', '').strip()
                    formatted_content += f"<h3>{heading}</h3>\n"
                elif para.startswith('#'):
                    heading = para.replace('#', '').strip()
                    formatted_content += f"<h2>{heading}</h2>\n"
                # Check if it's a list item
                elif para.startswith('- ') or para.startswith('* '):
                    items = para.split('\n')
                    formatted_content += "<ul>\n"
                    for item in items:
                        if item.strip().startswith(('- ', '* ')):
                            item_text = item.strip()[2:]
                            formatted_content += f"<li>{item_text}</li>\n"
                    formatted_content += "</ol>\n"
                # Regular paragraph
                else:
                    # Bold text formatting
                    para = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para)
                    # Italic text formatting
                    para = re.sub(r'\*(.*?)\*', r'<em>\1</em>', para)
                    formatted_content += f"<p>{para}</p>\n"
           
            return formatted_content
       
        # Create HTML content
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive Financial Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #1f4e79;
            --secondary-color: #2e86ab;
            --accent-color: #f18f01;
            --danger-color: #e74c3c;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --light-bg: #f8f9fa;
            --dark-bg: #2c3e50;
            --text-color: #2c3e50;
            --border-color: #e0e0e0;
        }}
       
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
       
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text-color);
            line-height: 1.6;
            background-color: var(--light-bg);
        }}
       
        .header {{
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
       
        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 300;
        }}
       
        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.9;
        }}
       
        .header .report-meta {{
            margin-top: 1rem;
            font-size: 0.9rem;
            opacity: 0.8;
        }}
       
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 2rem;
        }}
       
        .nav-tabs {{
            display: flex;
            background: white;
            border-bottom: 3px solid var(--primary-color);
            margin: 2rem 0 0 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
       
        .nav-tab {{
            padding: 1rem 2rem;
            background: none;
            border: none;
            cursor: pointer;
            font-size: 1rem;
            color: var(--text-color);
            border-bottom: 3px solid transparent;
            transition: all 0.3s ease;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
       
        .nav-tab:hover {{
            background-color: var(--light-bg);
            color: var(--primary-color);
        }}
       
        .nav-tab.active {{
            color: var(--primary-color);
            border-bottom-color: var(--primary-color);
            background-color: var(--light-bg);
            font-weight: 600;
        }}
       
        .tab-content {{
            display: none;
            padding: 2rem 0;
            animation: fadeIn 0.5s ease-in;
        }}
       
        .tab-content.active {{
            display: block;
        }}
       
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
       
        .card {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }}
       
        .card-header {{
            display: flex;
            align-items: center;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid var(--light-bg);
        }}
       
        .card-header h2 {{
            color: var(--primary-color);
            font-size: 1.8rem;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
       
        .grid {{
            display: grid;
            gap: 2rem;
        }}
       
        .grid-2 {{ grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); }}
        .grid-3 {{ grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); }}
        .grid-4 {{ grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); }}
       
        .metric-card {{
            background: linear-gradient(135deg, var(--secondary-color), var(--primary-color));
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
       
        .metric-card h3 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            font-weight: 300;
        }}
       
        .metric-card p {{
            font-size: 1.1rem;
            opacity: 0.9;
        }}
       
        .table-container {{
            overflow-x: auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
       
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}
       
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
       
        th {{
            background-color: var(--primary-color);
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
       
        tr:hover {{
            background-color: var(--light-bg);
        }}
       
        .chart-container {{
            background: white;
            border-radius: 12px;
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }}
       
        .chart-title {{
            text-align: center;
            color: var(--primary-color);
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--light-bg);
        }}
       
        .content-section {{
            background: white;
            border-radius: 12px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            line-height: 1.8;
        }}
       
        .content-section h3 {{
            color: var(--primary-color);
            font-size: 1.4rem;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid var(--accent-color);
            padding-left: 1rem;
        }}
       
        .content-section p {{
            margin-bottom: 1rem;
            text-align: justify;
        }}
       
        .content-section ul, .content-section ol {{
            margin: 1rem 0;
            padding-left: 2rem;
        }}
       
        .content-section li {{
            margin-bottom: 0.5rem;
        }}
       
        .alert {{
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
            border-left: 4px solid;
        }}
       
        .alert-info {{
            background-color: #e3f2fd;
            border-color: var(--secondary-color);
            color: #1565c0;
        }}
       
        .alert-warning {{
            background-color: #fff3e0;
            border-color: var(--warning-color);
            color: #e65100;
        }}
       
        .alert-danger {{
            background-color: #ffebee;
            border-color: var(--danger-color);
            color: #c62828;
        }}
       
        .alert-success {{
            background-color: #e8f5e8;
            border-color: var(--success-color);
            color: #2e7d32;
        }}
       
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            font-weight: 600;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
       
        .badge-primary {{ background-color: var(--primary-color); color: white; }}
        .badge-secondary {{ background-color: var(--secondary-color); color: white; }}
        .badge-success {{ background-color: var(--success-color); color: white; }}
        .badge-warning {{ background-color: var(--warning-color); color: white; }}
        .badge-danger {{ background-color: var(--danger-color); color: white; }}
       
        .footer {{
            background-color: var(--dark-bg);
            color: white;
            text-align: center;
            padding: 2rem 0;
            margin-top: 4rem;
        }}
       
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
       
        .stat-item {{
            text-align: center;
            padding: 1rem;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
       
        .stat-item .number {{
            font-size: 2rem;
            font-weight: 600;
            color: var(--primary-color);
        }}
       
        .stat-item .label {{
            font-size: 0.9rem;
            color: #666;
            margin-top: 0.5rem;
        }}
       
        .document-list {{
            display: grid;
            gap: 1rem;
            margin-top: 1rem;
        }}
       
        .document-item {{
            background: var(--light-bg);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid var(--accent-color);
        }}
       
        .document-item h4 {{
            color: var(--primary-color);
            margin-bottom: 0.5rem;
        }}
       
        @media (max-width: 768px) {{
            .container {{ padding: 0 1rem; }}
            .header h1 {{ font-size: 2rem; }}
            .card {{ padding: 1rem; }}
            .nav-tab {{ padding: 0.75rem 1rem; }}
            .grid-2, .grid-3, .grid-4 {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <header class="header">
        <div class="container">
            <h1><i class="fas fa-chart-line"></i> Comprehensive Financial Analysis Report</h1>
            <div class="subtitle">AI-Powered Financial Intelligence & Strategic Insights</div>
            <div class="report-meta">
                Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')} |
                {len(set(inf['source_document'] for inf in self.all_inferences))} Documents Analyzed |
                {len(self.all_inferences)} Sections Processed
            </div>
        </div>
    </header>

    <div class="container">
        <nav class="nav-tabs">
            <button class="nav-tab active" onclick="showTab('overview')">
                <i class="fas fa-tachometer-alt"></i> Executive Overview
            </button>
            <button class="nav-tab" onclick="showTab('financial')">
                <i class="fas fa-dollar-sign"></i> Financial Metrics
            </button>
            <button class="nav-tab" onclick="showTab('insights')">
                <i class="fas fa-lightbulb"></i> Strategic Insights
            </button>
            <button class="nav-tab" onclick="showTab('risks')">
                <i class="fas fa-exclamation-triangle"></i> Risk Assessment
            </button>
            <button class="nav-tab" onclick="showTab('charts')">
                <i class="fas fa-chart-bar"></i> Visualizations
            </button>
            <button class="nav-tab" onclick="showTab('documents')">
                <i class="fas fa-file-alt"></i> Document Analysis
            </button>
            <button class="nav-tab" onclick="showTab('recommendations')">
                <i class="fas fa-tasks"></i> Recommendations
            </button>
        </nav>

        <!-- Executive Overview Tab -->
        <div id="overview" class="tab-content active">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="number">{len(set(inf['source_document'] for inf in self.all_inferences))}</div>
                    <div class="label">Documents Analyzed</div>
                </div>
                <div class="stat-item">
                    <div class="number">{len(self.all_inferences)}</div>
                    <div class="label">Sections Processed</div>
                </div>
                <div class="stat-item">
                    <div class="number">{len(self.financial_data)}</div>
                    <div class="label">Financial Metrics</div>
                </div>
                <div class="stat-item">
                    <div class="number">{sum(len(inf.get('insights', [])) for inf in self.all_inferences)}</div>
                    <div class="label">Insights Generated</div>
                </div>
                <div class="stat-item">
                    <div class="number">{sum(len(inf.get('recommendations', [])) for inf in self.all_inferences)}</div>
                    <div class="label">Recommendations</div>
                </div>
                <div class="stat-item">
                    <div class="number">{sum(len(inf.get('concerns', [])) for inf in self.all_inferences)}</div>
                    <div class="label">Risk Factors</div>
                </div>
            </div>

            <div class="content-section">
                <div class="card-header">
                    <h2><i class="fas fa-crown"></i> Executive Summary</h2>
                </div>
                <div class="alert alert-info">
                    <strong>Report Scope:</strong> This comprehensive analysis covers {len(set(inf['source_document'] for inf in self.all_inferences))} financial documents
                    including regulatory filings, management reports, and strategic documents.
                </div>
                {format_content_for_html(exec_summary)}
            </div>
        </div>

        <!-- Financial Metrics Tab -->
        <div id="financial" class="tab-content">
            <div class="grid grid-3">
                <div class="metric-card">
                    <h3>{len(self.financial_data)}</h3>
                    <p>Total Financial Metrics Identified</p>
                </div>
                <div class="metric-card">
                    <h3>{sum(len(values) for values in self.financial_data.values())}</h3>
                    <p>Total Data Points</p>
                </div>
                <div class="metric-card">
                    <h3>{len([m for m in self.financial_data.keys() if 'revenue' in m.lower() or 'profit' in m.lower()])}</h3>
                    <p>Revenue & Profit Metrics</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2><i class="fas fa-table"></i> Key Financial Metrics</h2>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Metric</th>
                                <th>Latest Value</th>
                                <th>Source Document</th>
                                <th>Section</th>
                                <th>Category</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([f'''
                            <tr>
                                <td><strong>{metric['metric']}</strong></td>
                                <td>{metric['value']}</td>
                                <td>{metric['source'][:40]}{'...' if len(metric['source']) > 40 else ''}</td>
                                <td>{metric['section'][:30]}{'...' if len(metric['section']) > 30 else ''}</td>
                                <td>
                                    <span class="badge {'badge-primary' if 'revenue' in metric['metric'].lower() else 'badge-secondary' if 'cost' in metric['metric'].lower() else 'badge-success'}">
                                        {'Revenue' if 'revenue' in metric['metric'].lower() else 'Cost' if 'cost' in metric['metric'].lower() else 'Other'}
                                    </span>
                                </td>
                            </tr>
                            ''' for metric in financial_metrics_summary[:50]])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Strategic Insights Tab -->
        <div id="insights" class="tab-content">
            <div class="content-section">
                <div class="card-header">
                    <h2><i class="fas fa-brain"></i> Strategic Intelligence Analysis</h2>
                </div>
                <div class="alert alert-success">
                    <strong>Analysis Depth:</strong> {sum(len(inf.get('insights', [])) for inf in self.all_inferences)} insights and
                    {sum(len(inf.get('strategic_implications', [])) for inf in self.all_inferences)} strategic implications identified.
                </div>
                {format_content_for_html(insights_analysis)}
            </div>

            <div class="card">
                <div class="card-header">
                    <h2><i class="fas fa-list"></i> Top Strategic Insights</h2>
                </div>
                <div class="document-list">
                    {''.join([f'''
                    <div class="document-item">
                        <h4>Strategic Insight #{i+1}</h4>
                        <p>{insight}</p>
                    </div>
                    ''' for i, insight in enumerate(list(set([insight for inf in self.all_inferences for insight in inf.get('insights', [])]))[:20])])}
                </div>
            </div>
        </div>

        <!-- Risk Assessment Tab -->
        <div id="risks" class="tab-content">
            <div class="grid grid-4">
                <div class="metric-card" style="background: linear-gradient(135deg, var(--danger-color), #c0392b);">
                    <h3>{sum(len(inf.get('concerns', [])) for inf in self.all_inferences)}</h3>
                    <p>Total Risk Factors</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, var(--warning-color), #d68910);">
                    <h3>{len(set([concern for inf in self.all_inferences for concern in inf.get('concerns', [])]))}</h3>
                    <p>Unique Risks</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));">
                    <h3>{len([inf for inf in self.all_inferences if inf.get('risk_assessment')])}</h3>
                    <p>Risk Assessments</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, #8e44ad, #9b59b6);">
                    <h3>{sum(len(inf.get('compliance_notes', [])) for inf in self.all_inferences)}</h3>
                    <p>Compliance Items</p>
                </div>
            </div>

            <div class="content-section">
                <div class="card-header">
                    <h2><i class="fas fa-shield-alt"></i> Comprehensive Risk Assessment</h2>
                </div>
                <div class="alert alert-warning">
                    <strong>Risk Overview:</strong> This assessment covers financial, operational, strategic, and compliance risks
                    identified across all analyzed documents.
                </div>
                {format_content_for_html(risk_assessment)}
            </div>
        </div>

        <!-- Visualizations Tab -->
        <div id="charts" class="tab-content">
            <div class="alert alert-info">
                <strong>Interactive Charts:</strong> All visualizations are interactive. Hover for details, click to drill down.
            </div>
           
            <div class="grid grid-2">
                {''.join([f'''
                <div class="chart-container">
                    <div class="chart-title">{chart.stem.replace('_', ' ').title()}</div>
                    <iframe src="{chart.name}" width="100%" height="500" frameborder="0"></iframe>
                </div>
                ''' for chart in charts if chart.exists()])}
            </div>
        </div>

        <!-- Document Analysis Tab -->
        <div id="documents" class="tab-content">
            <div class="card">
                <div class="card-header">
                    <h2><i class="fas fa-folder-open"></i> Document Analysis Summary</h2>
                </div>
                <div class="table-container">
                    <table>
                        <thead>
                            <tr>
                                <th>Document</th>
                                <th>Sections</th>
                                <th>Insights</th>
                                <th>Concerns</th>
                                <th>Recommendations</th>
                                <th>Priority</th>
                            </tr>
                        </thead>
                        <tbody>
                            {''.join([f'''
                            <tr>
                                <td><strong>{doc_name[:50]}{'...' if len(doc_name) > 50 else ''}</strong></td>
                                <td>{count}</td>
                                <td>{sum(len(inf.get('insights', [])) for inf in self.all_inferences if inf['source_document'] == doc_name)}</td>
                                <td>{sum(len(inf.get('concerns', [])) for inf in self.all_inferences if inf['source_document'] == doc_name)}</td>
                                <td>{sum(len(inf.get('recommendations', [])) for inf in self.all_inferences if inf['source_document'] == doc_name)}</td>
                                <td>
                                    <span class="badge {'badge-danger' if sum(len(inf.get('concerns', [])) for inf in self.all_inferences if inf['source_document'] == doc_name) > 5 else 'badge-warning' if sum(len(inf.get('concerns', [])) for inf in self.all_inferences if inf['source_document'] == doc_name) > 2 else 'badge-success'}">
                                        {'High' if sum(len(inf.get('concerns', [])) for inf in self.all_inferences if inf['source_document'] == doc_name) > 5 else 'Medium' if sum(len(inf.get('concerns', [])) for inf in self.all_inferences if inf['source_document'] == doc_name) > 2 else 'Low'}
                                    </span>
                                </td>
                            </tr>
                            ''' for doc_name, count in doc_counts.items()])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Recommendations Tab -->
        <div id="recommendations" class="tab-content">
            <div class="grid grid-3">
                <div class="metric-card" style="background: linear-gradient(135deg, var(--success-color), #229954);">
                    <h3>{sum(len(inf.get('recommendations', [])) for inf in self.all_inferences)}</h3>
                    <p>Total Recommendations</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, var(--secondary-color), #2471a3);">
                    <h3>{len(set([rec for inf in self.all_inferences for rec in inf.get('recommendations', [])]))}</h3>
                    <p>Unique Actions</p>
                </div>
                <div class="metric-card" style="background: linear-gradient(135deg, var(--accent-color), #d68910);">
                    <h3>{sum(len(inf.get('strategic_implications', [])) for inf in self.all_inferences)}</h3>
                    <p>Strategic Implications</p>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2><i class="fas fa-clipboard-check"></i> Priority Recommendations</h2>
                </div>
                <div class="document-list">
                    {''.join([f'''
                    <div class="document-item">
                        <h4>
                            <span class="badge {'badge-danger' if i < 5 else 'badge-warning' if i < 15 else 'badge-primary'}">
                                {'High Priority' if i < 5 else 'Medium Priority' if i < 15 else 'Standard Priority'}
                            </span>
                            Recommendation #{i+1}
                        </h4>
                        <p>{rec}</p>
                    </div>
                    ''' for i, rec in enumerate(list(set([rec for inf in self.all_inferences for rec in inf.get('recommendations', [])]))[:30])])}
                </div>
            </div>
        </div>
    </div>

    <footer class="footer">
        <div class="container">
            <p>&copy; {datetime.now().year} AI-Powered Financial Analysis Report | Generated with Advanced Analytics</p>
            <p>Report includes {len(self.all_inferences)} analyzed sections from {len(set(inf['source_document'] for inf in self.all_inferences))} documents</p>
        </div>
    </footer>

    <script>
        function showTab(tabName) {{
            // Hide all tab contents
            const tabContents = document.querySelectorAll('.tab-content');
            tabContents.forEach(content => {{
                content.classList.remove('active');
            }});

            // Remove active class from all nav tabs
            const navTabs = document.querySelectorAll('.nav-tab');
            navTabs.forEach(tab => {{
                tab.classList.remove('active');
            }});

            // Show selected tab content
            document.getElementById(tabName).classList.add('active');

            // Add active class to clicked nav tab (find by onclick attribute)
            const targetTab = document.querySelector(`[onclick="showTab('${{tabName}}')"]`);
            if (targetTab) {{
                targetTab.classList.add('active');
            }}
        }}

        // Initialize page
        document.addEventListener('DOMContentLoaded', function() {{
            // Add smooth scrolling
            document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                anchor.addEventListener('click', function (e) {{
                    e.preventDefault();
                    const target = document.querySelector(this.getAttribute('href'));
                    if (target) {{
                        target.scrollIntoView({{
                            behavior: 'smooth'
                        }});
                    }}
                }});
            }});

            // Add loading states for charts
            const chartContainers = document.querySelectorAll('.chart-container iframe');
            chartContainers.forEach(iframe => {{
                iframe.addEventListener('load', function() {{
                    this.style.opacity = '1';
                }});
                iframe.style.opacity = '0.5';
            }});
        }});
    </script>
</body>
</html>"""

        # Save HTML file
        html_path = self.output_folder / f"Financial_Analysis_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
       
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
       
        logger.info(f"HTML report created: {html_path}")
        return html_path
   
    def generate_report(self):
        """Main method to generate the complete report"""
        logger.info("=" * 60)
        logger.info("STARTING FIXED FINANCIAL REPORT GENERATION")
        logger.info("=" * 60)
       
        try:
            # Check if aggregated MD file already exists
            aggregated_md = self.output_folder / "aggregated_analysis.md"
           
            if aggregated_md.exists():
                logger.info("Found existing aggregated_analysis.md file - skipping document processing")
                logger.info("Proceeding directly to HTML report generation...")
            else:
                # Step 1: Process all documents with fixed parallel processing
                logger.info("Step 1: Processing documents with fixed parallel processing...")
                aggregated_md = self.process_all_documents_fixed()
           
            # Step 2: Create comprehensive visualizations
            logger.info("Step 2: Creating comprehensive visualizations...")
            charts = self.create_visualizations()
           
            # Step 3: Generate HTML report
            logger.info("Step 3: Generating comprehensive HTML report...")
            report_path = self.create_html_report(aggregated_md, charts)
           
            logger.info("=" * 60)
            logger.info("REPORT GENERATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"Output folder: {self.output_folder}")
            logger.info(f"HTML report: {report_path}")
            logger.info(f"Aggregated analysis: {aggregated_md}")
            logger.info(f"Charts created: {len(charts)}")
           
            return report_path
           
        except Exception as e:
            logger.error(f"Error during report generation: {str(e)}")
            raise
        finally:
            # Cleanup
            if hasattr(self, 'vllm_processor'):
                self.vllm_processor.cleanup()
   
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'vllm_processor'):
            self.vllm_processor.cleanup()


def main():
    """Main function to run the fixed report generator"""
    # Configuration
    INPUT_FOLDER = "pdf_report_generator/md_files"  # Folder containing markdown files
    OUTPUT_FOLDER = "pdf_report_generator/parallel_report"  # Output folder
   
    # Validate input folder
    if not Path(INPUT_FOLDER).exists():
        logger.error(f"Input folder not found: {INPUT_FOLDER}")
        return
   
    # Check if there are any markdown files
    md_files = list(Path(INPUT_FOLDER).glob("*.md"))
    if not md_files:
        logger.error(f"No markdown files found in {INPUT_FOLDER}")
        return
   
    logger.info(f"Found {len(md_files)} markdown files to process")
   
    try:
        # Create report generator
        generator = FixedFinancialReportGenerator(INPUT_FOLDER, OUTPUT_FOLDER)
       
        # Generate report
        report_path = generator.generate_report()
       
        logger.info(f"\n🎉 SUCCESS! Your comprehensive financial report is ready!")
        logger.info(f"📁 Report location: {report_path}")
        logger.info(f"🌐 Open the HTML file in your browser to view the interactive report")
       
    except Exception as e:
        logger.error(f"❌ Error generating report: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()