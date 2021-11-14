n_images <- 5000
#reports <- c("cudaapisum", "gpukernsum", "gpumemtimesum", "gpumemsizesum", "osrtsum", "nvtxsum")
reports <- c("osrtsum", "nvtxsum")


for(do_it_again in seq(1, 10, 1)){
    for(thread in c(2 ** seq(0, 4))){
        for(report in reports){

            in_report_name <- paste("CPU", thread, n_images, do_it_again, sep="_")
            out_report_name <- paste("CPU", thread, n_images, do_it_again, report, sep="_")
            print(paste0("nsys stats -q --report=", report, " -f csv ",
                         in_report_name, ".qdrep > ", out_report_name, ".csv"))

        }   
    }
}


for(do_it_again in seq(1, 10, 1)){    
    for(batch in c(1, 500, 1000, 1200)){
#        for(stream in c(2 ** seq(0, 4))){
        for(stream in c(16)){
            for(report in reports){
                in_report_name <- paste("GPU", batch, n_images,
                                         stream, do_it_again, sep="_")
                out_report_name <- paste("GPU", batch, n_images,
                                         stream, do_it_again, report, sep="_")
                print(paste0("nsys stats -q --report=", report, " -f csv ",
                             in_report_name, ".qdrep > ", out_report_name, ".csv"))
                
            }
        }
    }        
}
