# workflow_report.jl
#
# Compiles report.jmd to PDF (preferred) or HTML (fallback) using Weave.jl.
#
# Prerequisites:
#   PDF:  pdflatex/lualatex must be on PATH (sudo apt install texlive-full)
#         Pandoc must be on PATH (sudo apt install pandoc)
#   HTML: no external dependencies — always works
#
# Run the traces and group workflows first so the report has data to display.

using Weave, Dates

function main_report(; format::Symbol = :auto)
    println("\n--- Starting: Report Generation ---")

    input_jmd  = joinpath(pwd(), "report.jmd")
    date_str   = Dates.format(now(), "yyyy-mm-dd")

    if !isfile(input_jmd)
        error("report.jmd not found in $(pwd()). Run from the project root.")
    end

    # Check whether workflow results exist so the user gets a useful warning.
    group_csv = joinpath(pwd(), "Parameter_Files", "Group_Trace", "latest", "All_sim_data.csv")
    trace_dir = joinpath(pwd(), "Parameter_Files", "Trace", "latest")
    if !isfile(group_csv)
        @warn "Group analysis results not found at $group_csv.\n" *
              "Run `julia main.jl --workflow group` first for the full report.\n" *
              "The report will compile but statistical sections will be skipped."
    end
    if !isdir(trace_dir) || isempty(readdir(trace_dir))
        @warn "Individual trace results not found in $trace_dir.\n" *
              "Run `julia main.jl --workflow traces` first."
    end

    # Determine format
    chosen_format = if format == :pdf
        :pdf
    elseif format == :html
        :html
    else
        # Auto-detect: try PDF, fall back to HTML
        latex_ok = !isnothing(Sys.which("pdflatex")) || !isnothing(Sys.which("lualatex"))
        latex_ok ? :pdf : :html
    end

    if chosen_format == :pdf
        out_path = joinpath(pwd(), "Notebook", "AP_Model_Report_$(date_str).pdf")
        println("Compiling to PDF: $out_path")
        try
            weave(input_jmd, doctype="md2pdf", out_path=out_path)
            println("\nPDF report saved to: $out_path")
        catch e
            @warn "PDF compilation failed (LaTeX error?). Falling back to HTML.\n$e"
            chosen_format = :html
        end
    end

    if chosen_format == :html
        out_path = joinpath(pwd(), "Notebook", "AP_Model_Report_$(date_str).html")
        println("Compiling to HTML: $out_path")
        try
            weave(input_jmd, doctype="md2html", out_path=out_path)
            println("\nHTML report saved to: $out_path")
            println("Open in a browser or convert to PDF with:")
            println("  chromium --headless --print-to-pdf=$out_path $out_path")
        catch e
            println("\nReport compilation failed:")
            showerror(stdout, e, catch_backtrace())
        end
    end
end
