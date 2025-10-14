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
    "financial_metrics": {{
        "revenue": number_or_null,
        "expenses": number_or_null,
        "net_income": number_or_null,
        "profit_margin": number_or_null
    }},
    "payment_info": {{
        "payment_method": "string or null",
        "payment_terms": "string or null",
        "bank_details": "string or null"
    }},
    "notes": "additional info"
}}

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
def safe_format_amount(value, currency_symbol='₹', default='N/A'):
    """Safely format amounts"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return f"{currency_symbol}{float(value):,.2f}"
    except:
        return default

def safe_get(data, key, default='N/A'):
    """Safely get dictionary value"""
    value = data.get(key)
    return value if value not in (None, '', 'null') else default

# -------------------------
# DataFrame Creation
# -------------------------
def create_dataframes(groq_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create DataFrames from extracted data"""
    
    # Line items DataFrame
    line_items = groq_data.get('line_items', [])
    if line_items:
        df_items = pd.DataFrame(line_items)
        df_items['document_type'] = groq_data.get('document_type', 'unknown')
        df_items['vendor_name'] = groq_data.get('vendor_name', 'Unknown')
        df_items['document_date'] = groq_data.get('document_date')
        df_items['currency'] = groq_data.get('currency', 'INR')
        
        for col in ['quantity', 'unit_price', 'amount']:
            if col in df_items.columns:
                df_items[col] = pd.to_numeric(df_items[col], errors='coerce').fillna(0)
    else:
        df_items = pd.DataFrame()
    
    # Summary DataFrame
    amounts = groq_data.get('amounts', {})
    df_summary = pd.DataFrame([{
        'document_type': groq_data.get('document_type', 'unknown'),
        'vendor_name': groq_data.get('vendor_name', 'Unknown'),
        'document_number': groq_data.get('document_number'),
        'document_date': groq_data.get('document_date'),
        'currency': groq_data.get('currency', 'INR'),
        'subtotal': amounts.get('subtotal', 0),
        'tax': amounts.get('tax', 0),
        'discount': amounts.get('discount', 0),
        'total': amounts.get('total', 0),
        'paid': amounts.get('paid', 0),
        'balance': amounts.get('balance', 0)
    }])
    
    for col in ['subtotal', 'tax', 'discount', 'total', 'paid', 'balance']:
        if col in df_summary.columns:
            df_summary[col] = pd.to_numeric(df_summary[col], errors='coerce').fillna(0)
    
    return df_items, df_summary

# -------------------------
# Dashboard
# -------------------------
def show_dashboard(groq_data: Dict, df_items: pd.DataFrame, df_summary: pd.DataFrame):
    """Display financial dashboard"""
    
    st.markdown("---")
    st.markdown("## 📊 Financial Analytics Dashboard")
    
    currency = groq_data.get('currency', 'INR')
    currency_symbol = '₹' if currency == 'INR' else ('$' if currency == 'USD' else currency)
    
    # Summary Metrics
    st.markdown("### 💰 Financial Summary")
    amounts = groq_data.get('amounts', {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = amounts.get('total', 0)
        st.metric("Total Amount", f"{currency_symbol}{total:,.2f}" if total else "N/A")
    with col2:
        tax = amounts.get('tax', 0)
        st.metric("Tax", f"{currency_symbol}{tax:,.2f}" if tax else "N/A")
    with col3:
        subtotal = amounts.get('subtotal', 0)
        st.metric("Subtotal", f"{currency_symbol}{subtotal:,.2f}" if subtotal else "N/A")
    with col4:
        balance = amounts.get('balance', 0)
        st.metric("Balance Due", f"{currency_symbol}{balance:,.2f}" if balance else "N/A")
    
    # Charts
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
    else:
        st.info("📊 No line items available for visualization")

# -------------------------
# Login Page
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
                    st.success("✅ Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("❌ Invalid username or password")
                    st.info("💡 Default credentials:\n- Username: **sfw**\n- Password: **admin**")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 14px;'>
            <p>🚀 Powered by Google Cloud Vision API + Groq AI</p>
            <p>🇮🇳 Optimized for Indian Financial Documents</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Main Application
# -------------------------
def main():
    st.set_page_config(
        page_title="SFW Financial Document OCR + AI",
        page_icon="₹",
        layout="wide"
    )
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # Show login page if not logged in
    if not st.session_state.logged_in:
        show_login_page()
        return
    
    # Main application (only shown after login)
    # Add logout button in sidebar
    with st.sidebar:
        st.markdown(f"**👤 Logged in as:** {st.session_state.username}")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.username = None
            st.rerun()
        st.markdown("---")
    
    st.title("₹ Universal Financial Document OCR + AI Analysis")
    st.markdown("**Handles:** Invoices • Receipts • P&L Statements • Balance Sheets • Financial Reports")
    
    # Initialize Vision API
    vision_client = get_vision_client()
    if not vision_client:
        st.error("⚠️ Google Cloud Vision API not configured")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Upload Document")
    uploaded_file = st.sidebar.file_uploader(
        "Upload financial document",
        type=['pdf', 'png', 'jpg', 'jpeg'],
        help="PDF or image (max 10MB)"
    )
    max_pages = st.sidebar.slider("Max PDF Pages", 1, 20, 10)
    
    # Session state
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'ocr_text' not in st.session_state:
        st.session_state.ocr_text = None
    
    # File processing
    if uploaded_file:
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size > MAX_FILE_SIZE_MB:
            st.error(f"File too large: {file_size:.1f}MB (max {MAX_FILE_SIZE_MB}MB)")
            return
        
        st.subheader("📤 Document Preview")
        
        file_ext = uploaded_file.name.split('.')[-1].lower()
        file_bytes = uploaded_file.getvalue()
        
        # Convert to images
        if file_ext == 'pdf':
            with st.spinner("Converting PDF..."):
                images = pdf_to_images(file_bytes, max_pages)
            if not images:
                return
            display_image = images[0]
        else:
            images = [Image.open(io.BytesIO(file_bytes))]
            display_image = images[0]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(display_image, caption="First Page", use_container_width=True)
        
        with col2:
            st.info(f"📄 **File:** {uploaded_file.name}\n\n📊 **Size:** {file_size:.2f} MB\n\n📑 **Pages:** {len(images)}")
            
            if st.button("🚀 Extract & Analyze", type="primary", use_container_width=True):
                # OCR
                with st.spinner("🔍 Running OCR..."):
                    all_text = []
                    progress = st.progress(0)
                    
                    for idx, img in enumerate(images):
                        progress.progress((idx + 1) / len(images))
                        img_bytes = prepare_image_for_vision(img)
                        ocr_result = extract_text_vision_api(img_bytes, vision_client)
                        
                        if 'error' in ocr_result:
                            st.error(f"OCR error page {idx + 1}: {ocr_result['error']}")
                            continue
                        
                        all_text.append(ocr_result['full_text'])
                    
                    combined_text = "\n\n".join(all_text)
                    st.session_state.ocr_text = combined_text
                    st.success("✅ OCR completed!")
                
                # AI Extraction
                with st.spinner("🤖 AI analyzing..."):
                    extracted_data = extract_financial_data_with_groq(combined_text)
                    
                    if extracted_data:
                        st.session_state.extracted_data = extracted_data
                        st.success("✅ Data extracted!")
                    else:
                        st.error("❌ Extraction failed")
    
    # Display results
    if st.session_state.extracted_data:
        data = st.session_state.extracted_data
        
        st.markdown("---")
        st.subheader("📊 Extracted Financial Data")
        
        doc_type = safe_get(data, 'document_type', 'Unknown').upper().replace('_', ' ')
        st.markdown(f"**Document Type:** `{doc_type}`")
        
        # Key metrics
        currency = data.get('currency', 'INR')
        currency_symbol = '₹' if currency == 'INR' else '$'
        amounts = data.get('amounts', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Amount", safe_format_amount(amounts.get('total'), currency_symbol))
        with col2:
            vendor = safe_get(data, 'vendor_name', 'N/A')
            st.metric("Vendor/Entity", vendor if len(vendor) < 30 else vendor[:27] + '...')
        with col3:
            st.metric("Document #", safe_get(data, 'document_number'))
        with col4:
            st.metric("Date", safe_get(data, 'document_date'))
        
        # Tabs
        st.markdown("## 📋 Detailed Information")
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Info", "💰 Amounts", "📦 Line Items", "🔍 Raw Data"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Document Details")
                st.json({
                    "Type": safe_get(data, 'document_type'),
                    "Number": safe_get(data, 'document_number'),
                    "Date": safe_get(data, 'document_date'),
                    "Due Date": safe_get(data, 'due_date'),
                    "Currency": currency
                })
            with col2:
                st.markdown("### Party Information")
                st.json({
                    "Vendor/Company": safe_get(data, 'vendor_name'),
                    "Customer": safe_get(data, 'customer_name'),
                    "Tax ID": safe_get(data, 'tax_id')
                })
        
        with tab2:
            st.markdown("### 💰 Financial Summary")
            amounts_df = pd.DataFrame([
                {"Category": "Subtotal", "Amount": safe_format_amount(amounts.get('subtotal'), currency_symbol)},
                {"Category": "Tax", "Amount": safe_format_amount(amounts.get('tax'), currency_symbol)},
                {"Category": "Discount", "Amount": safe_format_amount(amounts.get('discount'), currency_symbol)},
                {"Category": "Total", "Amount": safe_format_amount(amounts.get('total'), currency_symbol)},
                {"Category": "Paid", "Amount": safe_format_amount(amounts.get('paid'), currency_symbol)},
                {"Category": "Balance Due", "Amount": safe_format_amount(amounts.get('balance'), currency_symbol)}
            ])
            st.dataframe(amounts_df, use_container_width=True, hide_index=True)
            
            # Financial metrics
            metrics = data.get('financial_metrics', {})
            if any(metrics.values()):
                st.markdown("### 📈 Financial Metrics")
                metrics_df = pd.DataFrame([
                    {"Metric": "Revenue", "Value": safe_format_amount(metrics.get('revenue'), currency_symbol)},
                    {"Metric": "Expenses", "Value": safe_format_amount(metrics.get('expenses'), currency_symbol)},
                    {"Metric": "Net Income", "Value": safe_format_amount(metrics.get('net_income'), currency_symbol)},
                    {"Metric": "Profit Margin", "Value": f"{metrics.get('profit_margin', 0):.2f}%" if metrics.get('profit_margin') else 'N/A'}
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with tab3:
            line_items = data.get('line_items', [])
            if line_items:
                st.markdown(f"### 📦 Line Items ({len(line_items)} items)")
                df_items, df_summary = create_dataframes(data)
                
                if not df_items.empty:
                    st.dataframe(df_items, use_container_width=True, hide_index=True)
                    csv = df_items.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv,
                        "line_items.csv",
                        "text/csv",
                        use_container_width=True
                    )
            else:
                st.info("No line items found")
        
        with tab4:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Structured Data")
                st.json(data)
            with col2:
                st.markdown("### Raw OCR Text")
                if st.session_state.ocr_text:
                    st.text_area("", st.session_state.ocr_text, height=400, label_visibility="collapsed")
        
        # Dashboard
        if data.get('line_items') or data.get('amounts'):
            df_items, df_summary = create_dataframes(data)
            show_dashboard(data, df_items, df_summary)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <small>🚀 Powered by Google Cloud Vision API + Groq AI | 
        🇮🇳 Optimized for Indian Financial Documents (₹)</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
