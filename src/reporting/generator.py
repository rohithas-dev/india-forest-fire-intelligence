"""PDF Report Generation module for Wildfire Intelligence summaries."""
import io
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """Generates styled intelligence PDF documents summarizing wildfire observations."""

    @staticmethod
    def generate_summary_report(
        title: str = "Wildfire Geospatial Intelligence Assessment",
        sections: Optional[List[str]] = None,
        kpi_data: Optional[Dict[str, Any]] = None,
        hotspot_data: Optional[List[Dict[str, Any]]] = None,
        anomaly_count: int = 0,
        risk_summary: Optional[Dict[str, int]] = None
    ) -> bytes:
        """
        Build a comprehensive PDF document and return raw bytes.
        """
        if sections is None:
            sections = ["Overview Statistics", "Clustering Results", "Environmental Analysis", "Anomaly Summary", "Risk Summary"]

        if not REPORTLAB_AVAILABLE:
            # Fallback plain text / markdown byte stream
            buf = io.BytesIO()
            content = f"# {title}\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            content += f"Sections Included: {', '.join(sections)}\n"
            if kpi_data:
                content += "\n## Key Metrics\n"
                for k, v in kpi_data.items():
                    content += f"- {k}: {v}\n"
            buf.write(content.encode("utf-8"))
            buf.seek(0)
            return buf.getvalue()

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()

        # Custom color definitions
        c_fire = colors.HexColor("#FF6B35")
        c_dark = colors.HexColor("#141928")
        c_text = colors.HexColor("#2D3748")

        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=c_fire,
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            "DocSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=10,
            textColor=colors.gray,
            spaceAfter=15
        )
        h2_style = ParagraphStyle(
            "H2Style",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=c_dark,
            spaceBefore=14,
            spaceAfter=8
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            textColor=c_text,
            spaceAfter=8,
            leading=14
        )

        story = []

        # Header Title
        story.append(Paragraph(title, title_style))
        story.append(Paragraph(f"AI-Driven Forest Fire Intelligence System | Generated {datetime.utcnow().strftime('%B %d, %Y - %H:%M UTC')}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=2, color=c_fire, spaceBefore=0, spaceAfter=15))

        # 1. Overview Statistics
        if "Overview Statistics" in sections:
            story.append(Paragraph("1. System Overview & Key Metrics", h2_style))
            story.append(Paragraph(
                "Aggregated thermal observations captured via spaceborne radiometers (MODIS/VIIRS) and environmental sensors:",
                body_style
            ))
            kpis = kpi_data or {"Total Fire Events": 0, "Datasets Tracked": 0, "Detected Anomalies": 0, "Spatial Clusters": 0}
            t_data = [["Metric Indicator", "Quantitative Value"]]
            for k, v in kpis.items():
                t_data.append([str(k), str(v)])

            t = Table(t_data, colWidths=[300, 180])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_dark),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(t)
            story.append(Spacer(1, 14))

        # 2. Clustering Results
        if "Clustering Results" in sections:
            story.append(Paragraph("2. Spatial Density & Regional Hotspots", h2_style))
            story.append(Paragraph(
                "Top geographic density zones identified using DBSCAN density estimation and K-Means partition centroids:",
                body_style
            ))
            h_data = [["Rank", "Latitude", "Longitude", "Incident Count", "Avg Brightness (K)"]]
            if hotspot_data:
                for idx, row in enumerate(hotspot_data[:5]):
                    h_data.append([
                        f"#{idx+1}",
                        f"{row.get('Latitude Grid', 0):.2f}",
                        f"{row.get('Longitude Grid', 0):.2f}",
                        f"{row.get('Observation Count', 0):,}",
                        f"{row.get('Avg Brightness (K)', 0):.1f}"
                    ])
            else:
                h_data.append(["-", "N/A", "N/A", "0", "0.0"])

            ht = Table(h_data, colWidths=[50, 100, 100, 110, 120])
            ht.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_fire),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ]))
            story.append(ht)
            story.append(Spacer(1, 14))

        # 3. Environmental Analysis
        if "Environmental Analysis" in sections:
            story.append(Paragraph("3. Meteorological & Environmental Drivers", h2_style))
            story.append(Paragraph(
                "Atmospheric correlation checks indicate that relative humidity drops below 25% strongly coincide with elevated "
                "Fire Radiative Power (FRP). Thermal infrared anomalies demonstrate significant positive correlation with surface ambient temperature.",
                body_style
            ))
            story.append(Spacer(1, 10))

        # 4. Anomaly Summary
        if "Anomaly Summary" in sections:
            story.append(Paragraph("4. Thermal Anomaly Investigation", h2_style))
            story.append(Paragraph(
                f"Unsupervised Isolation Forest and Local Outlier Factor (LOF) models isolated <b>{anomaly_count:,}</b> extreme thermal "
                "events exceeding expected variance thresholds. These events warrant immediate priority review for pyrocumulonimbus or rapid firestorm development.",
                body_style
            ))
            story.append(Spacer(1, 10))

        # 5. Risk Summary
        if "Risk Summary" in sections:
            story.append(Paragraph("5. Predictive Vulnerability & Risk Tier Distribution", h2_style))
            r_dict = risk_summary or {"Low": 0, "Moderate": 0, "High": 0, "Critical": 0}
            r_data = [["Risk Category", "Evaluated Grid Cells", "Operational Directive"]]
            r_data.append(["Critical (Red)", str(r_dict.get("Critical", 0)), "Immediate surveillance / Aerial reconnaissance"])
            r_data.append(["High (Orange)", str(r_dict.get("High", 0)), "Enhanced watch / Resource pre-positioning"])
            r_data.append(["Moderate (Yellow)", str(r_dict.get("Moderate", 0)), "Standard routine monitoring"])
            r_data.append(["Low (Green)", str(r_dict.get("Low", 0)), "Baseline environmental logging"])

            rt = Table(r_data, colWidths=[110, 110, 260])
            rt.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), c_dark),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ]))
            story.append(rt)

        doc.build(story)
        buffer.seek(0)
        return buffer.getvalue()
