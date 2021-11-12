n_images <- 5000

for(do_it_again in seq(1, 1, 1)){
  for(thread in c(2 ** seq(0, 4))){
      out_report_name <- paste("CPU", thread, n_images, do_it_again, sep="_")
      print(out_report_name)
      system(paste0("nsys stats -q --report=nvtxsum -f csv ", out_report_name, ".qdrep > ", out_report_name, ".csv"))
  }
}


for(do_it_again in seq(1, 1, 1)){    
    for(batch in c(1, 500, 1000, 1200)){
        for(stream in c(2 ** seq(0, 4))){
            out_report_name <- paste("GPU", batch, n_images, stream, do_it_again, sep="_")
            print(out_report_name)
            system(paste0("nsys stats -q --report=nvtxsum -f csv ", out_report_name, ".qdrep > ", out_report_name, ".csv"))
        }
    }        
}
