
# streamlit_app.py
"""
Universal Financial Document OCR using Google Cloud Vision API + Groq AI
Handles: Invoices, P&L Statements, Receipts, Balance Sheets, Financial Reports
"""

import streamlit as st
from PIL import Image
import io
import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pdf2image import convert_from_bytes
from typing import Dict, List, Tuple
from google.cloud import vision
from google.oauth2 import service_account
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configuration
MAX_FILE_SIZE_MB = 10

# -------------------------
# Google Vision API
# -------------------------
@st.cache_resource
def get_vision_client():
    """Initialize Google Vision API client"""
    try:
        json_path = "august-terminus-462709-i3-e55bfc5c1c0f.json"
        if os.path.exists(json_path):
            credentials = service_account.Credentials.from_service_account_file(json_path)
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        if 'gcp_service_account' in st.secrets:
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
            return vision.ImageAnnotatorClient(credentials=credentials)
        
        return vision.ImageAnnotatorClient()
    except Exception as e:
        st.error(f"Vision API initialization failed: {e}")
        return None

# -------------------------
# Image Processing
# -------------------------
def prepare_image_for_vision(image: Image.Image, max_size: int = 4000) -> bytes:
    """Convert PIL Image to bytes for Vision API"""
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)
    
    if image.mode not in ('RGB', 'L'):
        image = image.convert('RGB')
    
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()

def pdf_to_images(pdf_bytes: bytes, max_pages: int = 10) -> List[Image.Image]:
    """Convert PDF to images"""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=max_pages, fmt='png')
        return images
    except Exception as e:
        st.error(f"PDF conversion failed: {e}")
        return []

# -------------------------
# OCR
# -------------------------
def extract_text_vision_api(image_bytes: bytes, client) -> Dict:
    """Extract text using Google Vision API"""
    if not client:
        return {"error": "Vision API client not initialized"}
    
    try:
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            return {"error": response.error.message}
        
        full_text = response.full_text_annotation.text if response.full_text_annotation else ""
        
        words = []
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        words.append({"text": word_text, "confidence": word.confidence})
        
        return {
            "full_text": full_text,
            "words": words,
            "avg_confidence": np.mean([w["confidence"] for w in words]) if words else 0
        }
    except Exception as e:
        return {"error": str(e)}

# -------------------------
# Groq AI Extraction
# -------------------------
def extract_financial_data_with_groq(text: str) -> Dict:
    """Extract structured data using Groq AI"""
    try:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY")
            except:
                pass
        
        if not groq_api_key:
            st.error("⚠️ GROQ_API_KEY not found in .env file or secrets")
            st.info("💡 Add GROQ_API_KEY=your_key to your .env file")
            return None
        
        client = Groq(api_key=groq_api_key)
        
        prompt = f"""Analyze this financial document and extract structured data in JSON format.

Document Text:
{text[:8000]}

For Balance Sheet or Financial Statement documents, extract ALL major categories with their values.
For P&L Statements, extract all revenue and expense line items.
For Invoices/Receipts, extract line items.

Return JSON with this structure (use null for missing data):
{{
    "document_type": "invoice|receipt|p&l_statement|balance_sheet|financial_statement|other",
    "vendor_name": "company name",
    "customer_name": "customer name if available",
    "document_number": "invoice/document number",
    "document_date": "YYYY-MM-DD",
    "due_date": "YYYY-MM-DD if available",
    "tax_id": "GST/tax ID",
    "currency": "INR|USD|EUR|GBP",
    "amounts": {{
        "subtotal": number_or_null,
        "tax": number_or_null,
        "discount": number_or_null,
        "total": number_or_null,
        "paid": number_or_null,
        "balance": number_or_null
    }},
    "line_items": [
        {{
            "description": "item description",
            "quantity": number_or_null,
            "unit_price": number_or_null,
            "amount": number
        }}
    ],
    "balance_sheet_items": [
        {{
            "category": "Assets|Liabilities|Equity",
            "subcategory": "Current Assets|Non-Current Assets|etc",
            "line_item": "specific item name",
            "current_year": number,
            "previous_year": number_or_null
        }}
    ],
    "pl_statement_items": [
        {{
            "category": "Revenue|Expenses",
            "line_item": "specific item name",
            "current_year": number,
            "previous_year": number_or_null
        }}
    ],
    "financial_metrics": {{
        "revenue": number_or_null,
        "total_income": number_or_null,
        "total_expenses": number_or_null,
        "net_income": number_or_null,
        "profit_margin": number_or_null,
        "total_assets": number_or_null,
        "total_liabilities": number_or_null,
        "total_equity": number_or_null
    }},
    "payment_info": {{
        "payment_method": "string or null",
        "payment_terms": "string or null",
        "bank_details": "string or null"
    }},
    "notes": "additional info"
}}

IMPORTANT: 
- For balance sheets, populate balance_sheet_items with ALL major line items
- For P&L statements, populate pl_statement_items with ALL revenue and expense items
- Extract actual numerical values from the document
- Use null only when data is truly missing

Return ONLY valid JSON, no explanations."""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Extract structured financial data. Return valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        
        result = response.choices[0].message.content
        
        if '```json' in result:
            result = result.split('```json')[1].split('```')[0].strip()
        elif '```' in result:
            result = result.split('```')[1].split('```')[0].strip()
        
        return json.loads(result)
        
    except Exception as e:
        st.error(f"Groq extraction error: {str(e)}")
        return None

# -------------------------
# Utility Functions
# -------------------------
def safe_format_amount(value, currency_symbol='₹'):
    """Safely format amounts"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return f"{currency_symbol}{float(value):,.2f}"
    except:
        return None

def safe_get(data, key, default=None):
    """Safely get dictionary value"""
    value = data.get(key)
    if value in (None, '', 'null'):
        return default
    return value

# -------------------------
# DataFrame Creation
# -------------------------
def create_dataframes(groq_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create DataFrames from extracted data"""
    
    # Line items DataFrame (for invoices/receipts)
    line_items = groq_data.get('line_items', [])
    if line_items:
        df_items = pd.DataFrame(line_items)
        df_items['document_type'] = groq_data.get('document_type')
        df_items['vendor_name'] = groq_data.get('vendor_name')
        df_items['document_date'] = groq_data.get('document_date')
        df_items['currency'] = groq_data.get('currency', 'INR')
        
        for col in ['quantity', 'unit_price', 'amount']:
            if col in df_items.columns:
                df_items[col] = pd.to_numeric(df_items[col], errors='coerce').fillna(0)
    else:
        df_items = pd.DataFrame()
    
    # Balance sheet DataFrame
    balance_sheet_items = groq_data.get('balance_sheet_items', [])
    if balance_sheet_items:
        df_balance = pd.DataFrame(balance_sheet_items)
        for col in ['current_year', 'previous_year']:
            if col in df_balance.columns:
                df_balance[col] = pd.to_numeric(df_balance[col], errors='coerce').fillna(0)
    else:
        df_balance = pd.DataFrame()
    
    # P&L statement DataFrame
    pl_items = groq_data.get('pl_statement_items', [])
    if pl_items:
        df_pl = pd.DataFrame(pl_items)
        for col in ['current_year', 'previous_year']:
            if col in df_pl.columns:
                df_pl[col] = pd.to_numeric(df_pl[col], errors='coerce').fillna(0)
    else:
        df_pl = pd.DataFrame()
    
    # Summary DataFrame
    amounts = groq_data.get('amounts', {})
    metrics = groq_data.get('financial_metrics', {})
    
    summary_data = {
        'document_type': groq_data.get('document_type'),
        'vendor_name': groq_data.get('vendor_name'),
        'document_number': groq_data.get('document_number'),
        'document_date': groq_data.get('document_date'),
        'currency': groq_data.get('currency', 'INR'),
        'subtotal': amounts.get('subtotal'),
        'tax': amounts.get('tax'),
        'discount': amounts.get('discount'),
        'total': amounts.get('total'),
        'paid': amounts.get('paid'),
        'balance': amounts.get('balance'),
        'revenue': metrics.get('revenue'),
        'total_income': metrics.get('total_income'),
        'total_expenses': metrics.get('total_expenses'),
        'net_income': metrics.get('net_income'),
        'total_assets': metrics.get('total_assets'),
        'total_liabilities': metrics.get('total_liabilities'),
        'total_equity': metrics.get('total_equity')
    }
    
    df_summary = pd.DataFrame([summary_data])
    
    for col in df_summary.columns:
        if col not in ['document_type', 'vendor_name', 'document_number', 'document_date', 'currency']:
            df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce')
    
    # Combine balance sheet and P&L into a unified financial statement dataframe
    df_financial = pd.DataFrame()
    if not df_balance.empty:
        df_financial = df_balance.copy()
        df_financial['statement_type'] = 'balance_sheet'
    if not df_pl.empty:
        df_pl_copy = df_pl.copy()
        df_pl_copy['statement_type'] = 'pl_statement'
        if 'subcategory' not in df_pl_copy.columns:
            df_pl_copy['subcategory'] = None
        df_financial = pd.concat([df_financial, df_pl_copy], ignore_index=True)
    
    return df_items, df_financial, df_summary

# -------------------------
# Dashboard
# -------------------------
def show_dashboard(groq_data: Dict, df_items: pd.DataFrame, df_financial: pd.DataFrame, df_summary: pd.DataFrame):
    """Display financial dashboard"""
    
    st.markdown("---")
    st.markdown("## 📊 Financial Analytics Dashboard")
    
    currency = groq_data.get('currency', 'INR')
    currency_symbol = '₹' if currency == 'INR' else ('$' if currency == 'USD' else '€' if currency == 'EUR' else '£' if currency == 'GBP' else currency)
    
    doc_type = groq_data.get('document_type', '').lower()
    
    # Summary Metrics based on document type
    st.markdown("### 💰 Financial Summary")
    
    if doc_type in ['balance_sheet', 'financial_statement']:
        metrics = groq_data.get('financial_metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_assets = metrics.get('total_assets')
            if total_assets:
                st.metric("Total Assets", f"{currency_symbol}{total_assets:,.2f}")
        with col2:
            total_liabilities = metrics.get('total_liabilities')
            if total_liabilities:
                st.metric("Total Liabilities", f"{currency_symbol}{total_liabilities:,.2f}")
        with col3:
            total_equity = metrics.get('total_equity')
            if total_equity:
                st.metric("Total Equity", f"{currency_symbol}{total_equity:,.2f}")
        with col4:
            net_income = metrics.get('net_income')
            if net_income:
                st.metric("Net Income", f"{currency_symbol}{net_income:,.2f}")
        
        # Balance Sheet Visualization
        if not df_financial.empty and 'current_year' in df_financial.columns:
            st.markdown("---")
            st.markdown("### 📈 Balance Sheet Analysis")
            
            # Filter and prepare data
            df_bs = df_financial[df_financial['statement_type'] == 'balance_sheet'].copy() if 'statement_type' in df_financial.columns else df_financial.copy()
            
            if not df_bs.empty and len(df_bs) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Top Items by Value (Current Year)")
                    top_items = df_bs.nlargest(10, 'current_year')
                    
                    fig1 = px.bar(
                        top_items,
                        x='current_year',
                        y='line_item',
                        orientation='h',
                        labels={'current_year': f'Amount ({currency_symbol})', 'line_item': 'Item'},
                        color='category' if 'category' in top_items.columns else None
                    )
                    fig1.update_traces(
                        texttemplate=f'{currency_symbol}%{{x:,.0f}}',
                        textposition='outside',
                        hovertemplate=f'<b>%{{y}}</b><br>Amount: {currency_symbol}%{{x:,.2f}}<extra></extra>'
                    )
                    fig1.update_layout(
                        yaxis={'categoryorder': 'total ascending'},
                        xaxis=dict(tickformat=',.0f'),
                        height=500,
                        showlegend=True if 'category' in top_items.columns else False
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown("#### Category Distribution")
                    
                    if 'category' in df_bs.columns:
                        category_totals = df_bs.groupby('category')['current_year'].sum().reset_index()
                        category_totals = category_totals[category_totals['current_year'] > 0]
                        
                        if not category_totals.empty:
                            fig2 = px.pie(
                                category_totals,
                                values='current_year',
                                names='category',
                                hole=0.4
                            )
                            fig2.update_traces(
                                texttemplate='%{label}<br>' + f'{currency_symbol}%{{value:,.0f}}',
                                textposition='inside',
                                hovertemplate=f'<b>%{{label}}</b><br>Amount: {currency_symbol}%{{value:,.2f}}<br>Percentage: %{{percent}}<extra></extra>'
                            )
                            fig2.update_layout(height=500, showlegend=True)
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        pie_data = df_bs.nlargest(10, 'current_year')
                        fig2 = px.pie(
                            pie_data,
                            values='current_year',
                            names='line_item',
                            hole=0.4
                        )
                        fig2.update_traces(
                            texttemplate='%{label}<br>' + f'{currency_symbol}%{{value:,.0f}}',
                            textposition='inside',
                            hovertemplate=f'<b>%{{label}}</b><br>Amount: {currency_symbol}%{{value:,.2f}}<br>Percentage: %{{percent}}<extra></extra>'
                        )
                        fig2.update_layout(height=500, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Year-over-year comparison
                if 'previous_year' in df_bs.columns and df_bs['previous_year'].sum() > 0:
                    st.markdown("---")
                    st.markdown("#### 📊 Year-over-Year Comparison")
                    
                    top_items_yoy = df_bs.nlargest(10, 'current_year')[['line_item', 'current_year', 'previous_year']].copy()
                    top_items_yoy['change'] = top_items_yoy['current_year'] - top_items_yoy['previous_year']
                    top_items_yoy['change_pct'] = ((top_items_yoy['current_year'] - top_items_yoy['previous_year']) / top_items_yoy['previous_year'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
                    
                    fig3 = go.Figure()
                    fig3.add_trace(go.Bar(
                        name='Previous Year',
                        x=top_items_yoy['line_item'],
                        y=top_items_yoy['previous_year'],
                        text=[f'{currency_symbol}{v:,.0f}' for v in top_items_yoy['previous_year']],
                        textposition='outside'
                    ))
                    fig3.add_trace(go.Bar(
                        name='Current Year',
                        x=top_items_yoy['line_item'],
                        y=top_items_yoy['current_year'],
                        text=[f'{currency_symbol}{v:,.0f}' for v in top_items_yoy['current_year']],
                        textposition='outside'
                    ))
                    fig3.update_layout(
                        barmode='group',
                        xaxis_title='Item',
                        yaxis_title=f'Amount ({currency_symbol})',
                        height=400,
                        xaxis={'tickangle': -45}
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
                # Summary Statistics
                st.markdown("---")
                st.markdown("### 📋 Summary Statistics")
                
                stats_df = pd.DataFrame({
                    'Metric': ['Total Items', 'Current Year Total', 'Previous Year Total', 'Highest Value', 'Lowest Value'],
                    'Value': [
                        f"{len(df_bs):,}",
                        f"{currency_symbol}{df_bs['current_year'].sum():,.2f}",
                        f"{currency_symbol}{df_bs['previous_year'].sum():,.2f}" if 'previous_year' in df_bs.columns else 'Not Available',
                        f"{currency_symbol}{df_bs['current_year'].max():,.2f}",
                        f"{currency_symbol}{df_bs['current_year'].min():,.2f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    elif doc_type in ['p&l_statement', 'pl_statement']:
        metrics = groq_data.get('financial_metrics', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            revenue = metrics.get('revenue') or metrics.get('total_income')
            if revenue:
                st.metric("Revenue", f"{currency_symbol}{revenue:,.2f}")
        with col2:
            expenses = metrics.get('total_expenses')
            if expenses:
                st.metric("Total Expenses", f"{currency_symbol}{expenses:,.2f}")
        with col3:
            net_income = metrics.get('net_income')
            if net_income:
                st.metric("Net Income", f"{currency_symbol}{net_income:,.2f}")
        with col4:
            profit_margin = metrics.get('profit_margin')
            if profit_margin:
                st.metric("Profit Margin", f"{profit_margin:.2f}%")
        
        # P&L Visualization
        if not df_financial.empty and 'current_year' in df_financial.columns:
            st.markdown("---")
            st.markdown("### 📈 P&L Analysis")
            
            df_pl = df_financial[df_financial['statement_type'] == 'pl_statement'].copy() if 'statement_type' in df_financial.columns else df_financial.copy()
            
            if not df_pl.empty and len(df_pl) > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Revenue vs Expenses")
                    
                    if 'category' in df_pl.columns:
                        category_totals = df_pl.groupby('category')['current_year'].sum().reset_index()
                        
                        fig1 = px.bar(
                            category_totals,
                            x='category',
                            y='current_year',
                            labels={'current_year': f'Amount ({currency_symbol})', 'category': 'Category'},
                            color='category'
                        )
                        fig1.update_traces(
                            texttemplate=f'{currency_symbol}%{{y:,.0f}}',
                            textposition='outside'
                        )
                        fig1.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    st.markdown("#### Expense Breakdown")
                    
                    expenses_df = df_pl[df_pl['category'] == 'Expenses'].copy() if 'category' in df_pl.columns else df_pl.copy()
                    
                    if not expenses_df.empty:
                        top_expenses = expenses_df.nlargest(8, 'current_year')
                        
                        fig2 = px.pie(
                            top_expenses,
                            values='current_year',
                            names='line_item',
                            hole=0.4
                        )
                        fig2.update_traces(
                            texttemplate='%{label}<br>' + f'{currency_symbol}%{{value:,.0f}}',
                            textposition='inside'
                        )
                        fig2.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)
    
    else:
        # Invoice/Receipt visualization
        amounts = groq_data.get('amounts', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total = amounts.get('total')
            if total:
                st.metric("Total Amount", f"{currency_symbol}{total:,.2f}")
        with col2:
            tax = amounts.get('tax')
            if tax:
                st.metric("Tax", f"{currency_symbol}{tax:,.2f}")
        with col3:
            subtotal = amounts.get('subtotal')
            if subtotal:
                st.metric("Subtotal", f"{currency_symbol}{subtotal:,.2f}")
        with col4:
            balance = amounts.get('balance')
            if balance:
                st.metric("Balance Due", f"{currency_symbol}{balance:,.2f}")
        
        # Charts for invoices
        if not df_items.empty and 'amount' in df_items.columns and 'description' in df_items.columns:
            st.markdown("---")
            st.markdown("### 📈 Visual Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Top 10 Items by Value")
                top_items = df_items.nlargest(10, 'amount')
                
                fig1 = px.bar(
                    top_items,
                    x='amount',
                    y='description',
                    orientation='h',
                    labels={'amount': f'Amount ({currency_symbol})', 'description': 'Item'}
                )
                fig1.update_traces(
                    texttemplate=f'{currency_symbol}%{{x:,.2f}}',
                    textposition='outside',
                    hovertemplate=f'<b>%{{y}}</b><br>Amount: {currency_symbol}%{{x:,.2f}}<extra></extra>'
                )
                fig1.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    xaxis=dict(tickformat=',.2f'),
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("#### Distribution")
                pie_data = df_items.nlargest(10, 'amount')
                
                fig2 = px.pie(
                    pie_data,
                    values='amount',
                    names='description',
                    hole=0.4
                )
                fig2.update_traces(
                    texttemplate='%{label}<br>' + f'{currency_symbol}%{{value:,.2f}}',
                    textposition='inside',
                    hovertemplate=f'<b>%{{label}}</b><br>Amount: {currency_symbol}%{{value:,.2f}}<br>Percentage: %{{percent}}<extra></extra>'
                )
                fig2.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Summary Statistics
            st.markdown("---")
            st.markdown("### 📋 Summary Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': ['Total Items', 'Total Amount', 'Average Amount', 'Highest Amount', 'Lowest Amount'],
                'Value': [
                    f"{len(df_items):,}",
                    f"{currency_symbol}{df_items['amount'].sum():,.2f}",
                    f"{currency_symbol}{df_items['amount'].mean():,.2f}",
                    f"{currency_symbol}{df_items['amount'].max():,.2f}",
                    f"{currency_symbol}{df_items['amount'].min():,.2f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

# -------------------------
# Login Page
# -------------------------
# -------------------------
# (Replace the tail of your file with the corrected login page closure + main)
# -------------------------

def show_login_page():
    """Display login page with SFW Technologies branding"""
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Display logo
        st.markdown("<div style='text-align: center; margin-bottom: 30px;'>", unsafe_allow_html=True)
        try:
            st.image(
                "https://softworkstech.com/wp-content/uploads/2022/09/softworklogo.svg",
                width=400
            )
        except:
            st.markdown("### 🏢 SFW Technologies")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Login form
        st.markdown("### 🔐 Login to Financial Document OCR")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit = st.form_submit_button("Login", use_container_width=True, type="primary")
            
            if submit:
                if username == "sfw" and password == "admin":
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success("✅ Login successful!")
                    # no explicit rerun call — Streamlit will re-run automatically on interaction
                else:
                    st.error("❌ Invalid username or password")
                    st.info("💡 Default credentials:\n- Username: **sfw**\n- Password: **admin**")
        
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>🚀 Powered by Google Cloud Vision API + Groq AI | 
        🇮🇳 Optimized for Indian Financial Documents (₹)</small>
    </div>
    """, unsafe_allow_html=True)



def main():
    """Main entrypoint for the Streamlit app"""
    st.set_page_config(page_title="Universal Financial Document OCR", layout="wide")
    
    # init session state
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = None
    
    # Simple nav: login vs app
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    # Logged-in app UI
    st.sidebar.header(f"Welcome, {st.session_state.username}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
    
    st.title("📥 Upload Financial Document (Invoice / P&L / Balance Sheet)")
    uploaded = st.file_uploader("Upload PDF / PNG / JPG (max 10MB)", type=['pdf', 'png', 'jpg', 'jpeg'], accept_multiple_files=False)
    
    if uploaded is None:
        st.info("Upload a document to begin OCR → Structured extraction → Dashboard")
        return
    
    # size guard
    uploaded.seek(0, os.SEEK_END)
    size_mb = uploaded.tell() / (1024 * 1024)
    uploaded.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File too large ({size_mb:.1f} MB). Max allowed: {MAX_FILE_SIZE_MB} MB.")
        return
    
    # convert to images (if pdf) or load image
    file_bytes = uploaded.read()
    file_type = uploaded.type if hasattr(uploaded, "type") else uploaded.name.split('.')[-1].lower()
    
    images = []
    if uploaded.name.lower().endswith('.pdf'):
        images = pdf_to_images(file_bytes, max_pages=10)
        if not images:
            st.error("Failed to convert PDF pages to images.")
            return
    else:
        try:
            pil_img = Image.open(io.BytesIO(file_bytes))
            images = [pil_img.convert('RGB')]
        except Exception as e:
            st.error(f"Failed to read image file: {e}")
            return
    
    # Initialize Vision client
    vision_client = get_vision_client()
    if vision_client is None:
        st.error("Vision API client could not be initialized. Check credentials.")
        return
    
    # OCR each page and concatenate text
    full_texts = []
    confidences = []
    words_all = []
    with st.spinner("Running OCR (Google Vision API)..."):
        for i, img in enumerate(images):
            try:
                img_bytes = prepare_image_for_vision(img)
                ocr_result = extract_text_vision_api(img_bytes, vision_client)
                if 'error' in ocr_result:
                    st.error(f"OCR error on page {i+1}: {ocr_result['error']}")
                else:
                    full_texts.append(ocr_result.get('full_text', ''))
                    confidences.append(ocr_result.get('avg_confidence', 0))
                    words_all.extend(ocr_result.get('words', []))
            except Exception as e:
                st.error(f"Unexpected OCR failure on page {i+1}: {e}")
    
    combined_text = "\n\n".join([t for t in full_texts if t])
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    
    st.markdown(f"**OCR Average Confidence:** {avg_conf:.2f}")
    st.expander("🔍 Show OCR text", expanded=False).write(combined_text[:20000] or "No text recognized")
    
    # Ask user to continue with Groq extraction
    if st.button("Extract structured financial data (Groq)"):
        with st.spinner("Extracting structured data with Groq AI..."):
            groq_data = extract_financial_data_with_groq(combined_text)
        
        if groq_data is None:
            st.error("Extraction failed or returned no data.")
            return
        
        # Show raw JSON and create dataframes
        st.subheader("Extracted JSON")
        st.json(groq_data)
        
        df_items, df_financial, df_summary = create_dataframes(groq_data)
        
        # Show dashboard
        show_dashboard(groq_data, df_items, df_financial, df_summary)
        
        # Data exports
        with st.expander("Download extracted tables"):
            if not df_items.empty:
                csv_items = df_items.to_csv(index=False).encode('utf-8')
                st.download_button("Download line items CSV", data=csv_items, file_name="line_items.csv", mime="text/csv")
            if not df_financial.empty:
                csv_fin = df_financial.to_csv(index=False).encode('utf-8')
                st.download_button("Download financials CSV", data=csv_fin, file_name="financials.csv", mime="text/csv")
            csv_summary = df_summary.to_csv(index=False).encode('utf-8')
            st.download_button("Download summary CSV", data=csv_summary, file_name="summary.csv", mime="text/csv")


if __name__ == "__main__":
    main()
