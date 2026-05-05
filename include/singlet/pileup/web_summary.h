#pragma once
// singlet-pileup: web_summary.h
// G-WEBSUMMARY: Generate a self-contained HTML web summary report, similar to
// Cell Ranger's web_summary.html. All CSS is inline; all charts are inline SVG.
// No external dependencies, no JavaScript required.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace singlet {

// ── Data structure ────────────────────────────────────────────────────────────

struct WebSummaryData {
    // Sample info
    std::string sample_name;
    std::string pipeline_version;
    std::string date;
    std::string organism;
    std::string protocol;

    // Key metrics
    uint64_t estimated_cells = 0;
    double   mean_reads_per_cell = 0.0;
    double   median_genes_per_cell = 0.0;
    uint64_t total_genes_detected = 0;
    double   median_umis_per_cell = 0.0;
    double   sequencing_saturation = 0.0;         // fraction [0,1]
    double   mapping_rate_genome = 0.0;            // fraction [0,1]
    double   mapping_rate_transcriptome = 0.0;     // fraction [0,1]
    double   mapping_rate_exonic = 0.0;            // fraction [0,1]
    double   mapping_rate_intronic = 0.0;          // fraction [0,1]
    double   mapping_rate_intergenic = 0.0;        // fraction [0,1]

    // Barcode rank data (sorted descending by UMI, all barcodes)
    std::vector<uint64_t> barcode_umi_counts;
    uint64_t cell_calling_threshold = 0;

    // Per-cell gene count distribution (called cells only)
    std::vector<uint32_t> genes_per_cell;

    // Warnings/alerts
    std::vector<std::string> warnings;
};

// ── Internal helpers ──────────────────────────────────────────────────────────

namespace detail_ws {

// HTML-escape text for safe embedding in attributes/text nodes
inline std::string html_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        if      (c == '&')  out += "&amp;";
        else if (c == '<')  out += "&lt;";
        else if (c == '>')  out += "&gt;";
        else if (c == '"')  out += "&quot;";
        else if (c == '\'') out += "&#39;";
        else                out += c;
    }
    return out;
}

inline std::string fmt_pct(double fraction) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f%%", fraction * 100.0);
    return buf;
}

inline std::string fmt_int_comma(uint64_t v) {
    std::string s = std::to_string(v);
    int n = static_cast<int>(s.size());
    std::string out;
    out.reserve(static_cast<size_t>(n + (n - 1) / 3));
    for (int i = 0; i < n; ++i) {
        if (i > 0 && (n - i) % 3 == 0) out += ',';
        out += s[static_cast<size_t>(i)];
    }
    return out;
}

inline std::string fmt_float1(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.1f", v);
    return buf;
}

// Build SVG knee plot (barcode rank plot, log-log)
// Downsamples to at most max_points for SVG size management.
inline std::string make_knee_plot_svg(const std::vector<uint64_t>& sorted_umis,
                                      uint64_t threshold,
                                      int svg_w = 500, int svg_h = 280) {
    if (sorted_umis.empty()) {
        // Empty placeholder
        std::ostringstream s;
        s << "<svg viewBox=\"0 0 " << svg_w << " " << svg_h
          << "\" xmlns=\"http://www.w3.org/2000/svg\">"
          << "<text x=\"" << svg_w/2 << "\" y=\"" << svg_h/2
          << "\" text-anchor=\"middle\" fill=\"#999\" font-size=\"13\">No data</text>"
          << "</svg>";
        return s.str();
    }

    // Margins
    const int ml = 58, mr = 18, mt = 18, mb = 46;
    const int pw = svg_w - ml - mr;  // plot width
    const int ph = svg_h - mt - mb;  // plot height

    // Log10 bounds
    const double log_x_min = 0.0;
    const double log_x_max = std::log10(static_cast<double>(sorted_umis.size()));
    const uint64_t max_umi = sorted_umis.front();
    const uint64_t min_umi = sorted_umis.back();
    const double log_y_max = std::log10(static_cast<double>(max_umi + 1));
    const double log_y_min = std::log10(static_cast<double>(std::max(uint64_t(1), min_umi)));

    const double log_x_range = (log_x_max > log_x_min) ? (log_x_max - log_x_min) : 1.0;
    const double log_y_range = (log_y_max > log_y_min) ? (log_y_max - log_y_min) : 1.0;

    auto to_px = [&](size_t idx, uint64_t umi) -> std::pair<double,double> {
        double lx = std::log10(static_cast<double>(idx + 1));
        double ly = std::log10(static_cast<double>(umi + 1));
        double px = ml + (lx - log_x_min) / log_x_range * pw;
        double py = mt + ph - (ly - log_y_min) / log_y_range * ph;
        return {px, py};
    };

    // Downsample: at most 800 points
    const size_t n = sorted_umis.size();
    const size_t max_pts = 800;
    std::vector<size_t> indices;
    if (n <= max_pts) {
        indices.resize(n);
        std::iota(indices.begin(), indices.end(), 0u);
    } else {
        indices.reserve(max_pts);
        for (size_t i = 0; i < max_pts; ++i)
            indices.push_back(static_cast<size_t>(i * (n - 1) / (max_pts - 1)));
    }

    // Split points into cells vs background by threshold
    std::ostringstream s;
    s << std::fixed << std::setprecision(2);
    s << "<svg viewBox=\"0 0 " << svg_w << " " << svg_h
      << "\" xmlns=\"http://www.w3.org/2000/svg\" font-family=\"Arial,sans-serif\">\n";

    // Grid lines (light gray)
    s << "<line x1=\"" << ml << "\" y1=\"" << mt
      << "\" x2=\"" << ml << "\" y2=\"" << (mt+ph)
      << "\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n";
    s << "<line x1=\"" << ml << "\" y1=\"" << (mt+ph)
      << "\" x2=\"" << (ml+pw) << "\" y2=\"" << (mt+ph)
      << "\" stroke=\"#e0e0e0\" stroke-width=\"1\"/>\n";
    // Horizontal grid lines
    for (int tick = 1; tick <= 4; ++tick) {
        double y = mt + ph - tick * ph / 4.0;
        s << "<line x1=\"" << ml << "\" y1=\"" << y
          << "\" x2=\"" << (ml+pw) << "\" y2=\"" << y
          << "\" stroke=\"#f0f0f0\" stroke-width=\"1\"/>\n";
    }

    // Build cell polyline (above threshold) and background polyline
    std::string cell_pts, bg_pts;
    bool sep_found = false;
    for (size_t idx : indices) {
        auto [px, py] = to_px(idx, sorted_umis[idx]);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.1f,%.1f ", px, py);
        if (threshold > 0 && sorted_umis[idx] >= threshold && !sep_found)
            cell_pts += buf;
        else {
            sep_found = true;
            bg_pts += buf;
        }
    }

    if (!bg_pts.empty())
        s << "<polyline points=\"" << bg_pts
          << "\" fill=\"none\" stroke=\"#b0bec5\" stroke-width=\"1.5\"/>\n";
    if (!cell_pts.empty())
        s << "<polyline points=\"" << cell_pts
          << "\" fill=\"none\" stroke=\"#1976d2\" stroke-width=\"2\"/>\n";

    // Threshold horizontal line
    if (threshold > 0) {
        double log_ty = std::log10(static_cast<double>(threshold + 1));
        double ty = mt + ph - (log_ty - log_y_min) / log_y_range * ph;
        s << "<line x1=\"" << ml << "\" y1=\"" << ty
          << "\" x2=\"" << (ml+pw) << "\" y2=\"" << ty
          << "\" stroke=\"#e53935\" stroke-width=\"1.5\" stroke-dasharray=\"5,3\"/>\n";
        s << "<text x=\"" << (ml+pw+2) << "\" y=\"" << (ty+4)
          << "\" font-size=\"9\" fill=\"#e53935\">threshold</text>\n";
    }

    // Axis labels — Y axis: a few log10 ticks
    s << "<text x=\"" << (ml-8) << "\" y=\"" << (mt + ph)
      << "\" font-size=\"10\" fill=\"#555\" text-anchor=\"end\">"
      << static_cast<int>(std::pow(10.0, log_y_min)) << "</text>\n";
    s << "<text x=\"" << (ml-8) << "\" y=\"" << mt
      << "\" font-size=\"10\" fill=\"#555\" text-anchor=\"end\">"
      << fmt_int_comma(max_umi) << "</text>\n";

    // X axis ticks
    s << "<text x=\"" << ml << "\" y=\"" << (mt+ph+14)
      << "\" font-size=\"10\" fill=\"#555\" text-anchor=\"middle\">1</text>\n";
    s << "<text x=\"" << (ml+pw) << "\" y=\"" << (mt+ph+14)
      << "\" font-size=\"10\" fill=\"#555\" text-anchor=\"middle\">"
      << fmt_int_comma(static_cast<uint64_t>(n)) << "</text>\n";

    // Axis captions
    s << "<text x=\"" << (ml + pw/2) << "\" y=\"" << (svg_h - 4)
      << "\" font-size=\"11\" fill=\"#333\" text-anchor=\"middle\">Barcode rank</text>\n";
    s << "<text transform=\"rotate(-90," << 11 << "," << (mt+ph/2)
      << ")\" x=\"11\" y=\"" << (mt+ph/2+4)
      << "\" font-size=\"11\" fill=\"#333\" text-anchor=\"middle\">UMI count</text>\n";

    // Legend
    s << "<rect x=\"" << (ml+10) << "\" y=\"" << (mt+8)
      << "\" width=\"12\" height=\"3\" fill=\"#1976d2\"/>\n";
    s << "<text x=\"" << (ml+26) << "\" y=\"" << (mt+14)
      << "\" font-size=\"9\" fill=\"#333\">Cells</text>\n";
    s << "<rect x=\"" << (ml+62) << "\" y=\"" << (mt+8)
      << "\" width=\"12\" height=\"3\" fill=\"#b0bec5\"/>\n";
    s << "<text x=\"" << (ml+78) << "\" y=\"" << (mt+14)
      << "\" font-size=\"9\" fill=\"#333\">Background</text>\n";

    s << "</svg>";
    return s.str();
}

// Build SVG histogram of genes per cell
inline std::string make_gene_histogram_svg(const std::vector<uint32_t>& genes_per_cell,
                                           int svg_w = 500, int svg_h = 220) {
    if (genes_per_cell.empty()) {
        std::ostringstream s;
        s << "<svg viewBox=\"0 0 " << svg_w << " " << svg_h
          << "\" xmlns=\"http://www.w3.org/2000/svg\">"
          << "<text x=\"" << svg_w/2 << "\" y=\"" << svg_h/2
          << "\" text-anchor=\"middle\" fill=\"#999\" font-size=\"13\">No data</text>"
          << "</svg>";
        return s.str();
    }

    const int ml = 52, mr = 14, mt = 16, mb = 42;
    const int pw = svg_w - ml - mr;
    const int ph = svg_h - mt - mb;

    // Build histogram bins (30 bins)
    const int n_bins = 30;
    uint32_t max_g = *std::max_element(genes_per_cell.begin(), genes_per_cell.end());
    uint32_t min_g = *std::min_element(genes_per_cell.begin(), genes_per_cell.end());
    if (max_g == min_g) max_g = min_g + 1;
    double bin_w_val = (max_g - min_g + 1.0) / n_bins;

    std::vector<uint32_t> counts(static_cast<size_t>(n_bins), 0);
    for (uint32_t g : genes_per_cell) {
        int bin = static_cast<int>((g - min_g) / bin_w_val);
        if (bin >= n_bins) bin = n_bins - 1;
        counts[static_cast<size_t>(bin)]++;
    }
    uint32_t max_count = *std::max_element(counts.begin(), counts.end());
    if (max_count == 0) max_count = 1;

    std::ostringstream s;
    s << std::fixed << std::setprecision(1);
    s << "<svg viewBox=\"0 0 " << svg_w << " " << svg_h
      << "\" xmlns=\"http://www.w3.org/2000/svg\" font-family=\"Arial,sans-serif\">\n";

    // Horizontal grid lines
    for (int t = 1; t <= 4; ++t) {
        double y = mt + ph - t * ph / 4.0;
        s << "<line x1=\"" << ml << "\" y1=\"" << y
          << "\" x2=\"" << (ml+pw) << "\" y2=\"" << y
          << "\" stroke=\"#f0f0f0\" stroke-width=\"1\"/>\n";
        uint32_t cnt_label = static_cast<uint32_t>(max_count * t / 4.0);
        s << "<text x=\"" << (ml-4) << "\" y=\"" << (y+4)
          << "\" font-size=\"9\" fill=\"#666\" text-anchor=\"end\">"
          << cnt_label << "</text>\n";
    }

    // Bars
    double bar_w = static_cast<double>(pw) / n_bins;
    for (int i = 0; i < n_bins; ++i) {
        double bar_h = ph * counts[static_cast<size_t>(i)] / static_cast<double>(max_count);
        double x = ml + i * bar_w;
        double y = mt + ph - bar_h;
        s << "<rect x=\"" << x << "\" y=\"" << y
          << "\" width=\"" << (bar_w - 1.0) << "\" height=\"" << bar_h
          << "\" fill=\"#42a5f5\" opacity=\"0.85\"/>\n";
    }

    // Axes
    s << "<line x1=\"" << ml << "\" y1=\"" << mt
      << "\" x2=\"" << ml << "\" y2=\"" << (mt+ph)
      << "\" stroke=\"#ccc\" stroke-width=\"1\"/>\n";
    s << "<line x1=\"" << ml << "\" y1=\"" << (mt+ph)
      << "\" x2=\"" << (ml+pw) << "\" y2=\"" << (mt+ph)
      << "\" stroke=\"#ccc\" stroke-width=\"1\"/>\n";

    // X axis labels — min and max
    s << "<text x=\"" << ml << "\" y=\"" << (mt+ph+14)
      << "\" font-size=\"9\" fill=\"#555\" text-anchor=\"middle\">" << min_g << "</text>\n";
    s << "<text x=\"" << (ml+pw) << "\" y=\"" << (mt+ph+14)
      << "\" font-size=\"9\" fill=\"#555\" text-anchor=\"middle\">" << max_g << "</text>\n";

    // Midpoint x label
    s << "<text x=\"" << (ml+pw/2) << "\" y=\"" << (mt+ph+14)
      << "\" font-size=\"9\" fill=\"#555\" text-anchor=\"middle\">"
      << (min_g + max_g) / 2 << "</text>\n";

    // Axis captions
    s << "<text x=\"" << (ml+pw/2) << "\" y=\"" << (svg_h-4)
      << "\" font-size=\"11\" fill=\"#333\" text-anchor=\"middle\">Genes per cell</text>\n";
    s << "<text transform=\"rotate(-90," << 11 << "," << (mt+ph/2)
      << ")\" x=\"11\" y=\"" << (mt+ph/2+4)
      << "\" font-size=\"11\" fill=\"#333\" text-anchor=\"middle\">Cells</text>\n";

    s << "</svg>";
    return s.str();
}

}  // namespace detail_ws

// ── Public API ─────────────────────────────────────────────────────────────────

/// Generate a self-contained HTML web summary report string.
/// The returned string is a complete HTML document with no external dependencies.
inline std::string generate_web_summary(const WebSummaryData& d) {
    using detail_ws::html_escape;
    using detail_ws::fmt_pct;
    using detail_ws::fmt_int_comma;
    using detail_ws::fmt_float1;

    std::ostringstream html;

    // ── Inline CSS ─────────────────────────────────────────────────────────────
    const char* css = R"css(
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Arial,Helvetica,sans-serif;background:#f5f7fa;color:#222;font-size:14px}
.header{background:linear-gradient(135deg,#1565c0,#1976d2);color:#fff;padding:18px 28px 14px}
.header h1{font-size:1.5em;font-weight:700;letter-spacing:.02em}
.header .meta{margin-top:6px;font-size:.85em;opacity:.88;display:flex;gap:24px;flex-wrap:wrap}
.header .meta span{display:flex;align-items:center;gap:4px}
.container{max-width:1100px;margin:22px auto;padding:0 16px}
.section-title{font-size:1em;font-weight:700;color:#1565c0;text-transform:uppercase;
  letter-spacing:.06em;margin:22px 0 10px;border-bottom:2px solid #1976d2;padding-bottom:4px}
.cards{display:flex;flex-wrap:wrap;gap:12px;margin-bottom:10px}
.card{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.10);
  padding:14px 18px;flex:1 1 180px;min-width:160px;max-width:240px}
.card .label{font-size:.78em;color:#666;margin-bottom:4px;font-weight:600;
  text-transform:uppercase;letter-spacing:.04em}
.card .value{font-size:1.55em;font-weight:700;color:#1565c0;line-height:1.1}
.card .value.warn{color:#e65100}
.chart-row{display:flex;flex-wrap:wrap;gap:16px;margin-bottom:4px}
.chart-box{background:#fff;border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.10);
  padding:14px 16px;flex:1 1 440px;min-width:340px}
.chart-box h3{font-size:.85em;font-weight:700;color:#444;margin-bottom:8px;
  text-transform:uppercase;letter-spacing:.04em}
.chart-box svg{display:block;width:100%;max-width:500px}
table{width:100%;border-collapse:collapse;background:#fff;
  border-radius:8px;box-shadow:0 1px 4px rgba(0,0,0,.10);overflow:hidden;margin-bottom:4px}
th{background:#e3f2fd;color:#1565c0;font-size:.82em;font-weight:700;padding:9px 14px;
   text-align:left;text-transform:uppercase;letter-spacing:.04em}
td{padding:8px 14px;border-bottom:1px solid #f0f0f0;font-size:.93em}
tr:last-child td{border-bottom:none}
tr:hover td{background:#fafcff}
.warn-box{background:#fff3e0;border-left:4px solid #ff9800;border-radius:4px;
  padding:10px 14px;margin-bottom:8px;font-size:.88em}
.warn-box .warn-title{font-weight:700;color:#e65100;margin-bottom:2px}
.no-warnings{color:#43a047;background:#e8f5e9;border-radius:4px;padding:8px 14px;
  font-size:.88em;border-left:4px solid #43a047;margin-bottom:8px}
.footer{text-align:center;color:#aaa;font-size:.78em;margin:28px 0 12px}
)css";

    // ── Document start ─────────────────────────────────────────────────────────
    html << "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
         << "<meta charset=\"UTF-8\">\n"
         << "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">\n"
         << "<title>singlify — " << html_escape(d.sample_name) << "</title>\n"
         << "<style>" << css << "</style>\n"
         << "</head>\n<body>\n";

    // ── Header ─────────────────────────────────────────────────────────────────
    html << "<div class=\"header\">\n"
         << "  <h1>&#128202; " << html_escape(d.sample_name) << "</h1>\n"
         << "  <div class=\"meta\">\n"
         << "    <span>&#128337; <b>Date:</b>&nbsp;" << html_escape(d.date) << "</span>\n"
         << "    <span>&#129516; <b>Organism:</b>&nbsp;" << html_escape(d.organism) << "</span>\n"
         << "    <span>&#128204; <b>Protocol:</b>&nbsp;" << html_escape(d.protocol) << "</span>\n"
         << "    <span>&#128736; <b>Pipeline:</b>&nbsp;singlify " << html_escape(d.pipeline_version) << "</span>\n"
         << "  </div>\n"
         << "</div>\n";

    html << "<div class=\"container\">\n";

    // ── Warnings ───────────────────────────────────────────────────────────────
    if (!d.warnings.empty()) {
        html << "<div class=\"section-title\">&#9888; Alerts</div>\n";
        for (const auto& w : d.warnings) {
            html << "<div class=\"warn-box\"><div class=\"warn-title\">Warning</div>"
                 << html_escape(w) << "</div>\n";
        }
    } else {
        html << "<div class=\"section-title\">Status</div>\n"
             << "<div class=\"no-warnings\">&#10003; No warnings — all quality metrics within expected ranges.</div>\n";
    }

    // ── Key Metrics ────────────────────────────────────────────────────────────
    html << "<div class=\"section-title\">Key Metrics</div>\n"
         << "<div class=\"cards\">\n";

    auto card = [&](const std::string& label, const std::string& value, bool warn = false) {
        html << "  <div class=\"card\">"
             << "<div class=\"label\">" << label << "</div>"
             << "<div class=\"value" << (warn ? " warn" : "") << "\">" << value << "</div>"
             << "</div>\n";
    };

    card("Estimated Cells",              fmt_int_comma(d.estimated_cells));
    card("Mean Reads / Cell",            fmt_int_comma(static_cast<uint64_t>(d.mean_reads_per_cell)));
    card("Median Genes / Cell",          fmt_float1(d.median_genes_per_cell));
    card("Total Genes Detected",         fmt_int_comma(d.total_genes_detected));
    card("Median UMI / Cell",            fmt_float1(d.median_umis_per_cell));
    card("Sequencing Saturation",        fmt_pct(d.sequencing_saturation),
         d.sequencing_saturation < 0.30);
    card("Reads Mapped to Genome",       fmt_pct(d.mapping_rate_genome),
         d.mapping_rate_genome < 0.50);
    card("Reads Mapped to Transcriptome",fmt_pct(d.mapping_rate_transcriptome),
         d.mapping_rate_transcriptome < 0.30);

    html << "</div>\n";  // cards

    // ── Charts ─────────────────────────────────────────────────────────────────
    html << "<div class=\"section-title\">Quality Charts</div>\n"
         << "<div class=\"chart-row\">\n";

    // Knee plot
    html << "  <div class=\"chart-box\">\n"
         << "    <h3>Barcode Rank Plot (Knee Plot)</h3>\n"
         << detail_ws::make_knee_plot_svg(d.barcode_umi_counts, d.cell_calling_threshold)
         << "\n  </div>\n";

    // Gene histogram
    html << "  <div class=\"chart-box\">\n"
         << "    <h3>Gene Distribution (Called Cells)</h3>\n"
         << detail_ws::make_gene_histogram_svg(d.genes_per_cell)
         << "\n  </div>\n";

    html << "</div>\n";  // chart-row

    // ── Mapping Summary Table ───────────────────────────────────────────────────
    html << "<div class=\"section-title\">Mapping Summary</div>\n"
         << "<table>\n"
         << "<thead><tr><th>Metric</th><th>Value</th></tr></thead>\n"
         << "<tbody>\n";

    auto trow = [&](const std::string& metric, const std::string& value) {
        html << "<tr><td>" << metric << "</td><td><b>" << value << "</b></td></tr>\n";
    };

    trow("Reads Mapped to Genome",             fmt_pct(d.mapping_rate_genome));
    trow("Reads Mapped Confidently to Transcriptome", fmt_pct(d.mapping_rate_transcriptome));
    trow("Reads Mapped to Exonic Regions",     fmt_pct(d.mapping_rate_exonic));
    trow("Reads Mapped to Intronic Regions",   fmt_pct(d.mapping_rate_intronic));
    trow("Reads Mapped to Intergenic Regions", fmt_pct(d.mapping_rate_intergenic));

    html << "</tbody></table>\n";

    // ── Footer ─────────────────────────────────────────────────────────────────
    html << "<div class=\"footer\">Generated by singlify "
         << html_escape(d.pipeline_version)
         << " &mdash; " << html_escape(d.date) << "</div>\n";

    html << "</div>\n";  // container
    html << "</body>\n</html>\n";

    return html.str();
}

/// Write a self-contained HTML web summary to the given file path.
/// Creates parent directories if needed. Does nothing if path is empty.
inline void write_web_summary(const std::string& path, const WebSummaryData& data) {
    if (path.empty()) return;
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "[web_summary] WARNING: cannot open " << path << " for writing\n";
        return;
    }
    f << generate_web_summary(data);
    std::cerr << "[web_summary] " << path << "\n";
}

}  // namespace singlet
