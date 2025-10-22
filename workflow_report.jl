# workflow_report.jl

# These packages are required for building the report
using Weave
using Dates

function main_report()
    println("\n--- Starting: Report Generation Workflow ---")
    
    # Define the input and output filenames
    input_jmd = "report.jmd"
    date_str = Dates.format(now(), "yyyy-mm-dd")
    output_filename = "AP_Model_Report_$(date_str).pdf"
    
    # Use the `notebook_folder` variable from config.jl to define the output path
    full_output_path = joinpath(notebook_folder, output_filename)

    println("Weaving $(input_jmd) to PDF...")
    try
        weave(input_jmd,
              doctype = "md2pdf",
              out_path = full_output_path)
        
        println("\nReport generation complete!")
        println("PDF saved to: $(full_output_path)")

    catch e
        println("\nERROR during report generation.")
        println("This can happen if LaTeX is not installed or if there's an error in the .jmd file.")
        showerror(stdout, e)
        println()
    end
end
