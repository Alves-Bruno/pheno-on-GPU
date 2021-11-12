#library(tidyverse)

cpu_bin <- "/home/users/bsalves/pheno-on-GPU/CPU-decode/cpu-decode"
gpu_bin <- "/home/users/bsalves/pheno-on-GPU/GPU-decode/project/run"
images <- "/tmp/2019/"
n_images <- 5000

nsys <- "nsys profile"
nsys_args <- " --force-overwrite true --export=json -o "

flush <- "sudo /sbin/sysctl vm.drop_caches=3"
#for(do_it_again in seq(1, 10, 1)){
for(do_it_again in seq(1, 1, 1)){
  for(thread in c(2 ** seq(0, 4))){
      print(paste0("export OMP_NUM_THREADS=", thread))
      system(paste0("export OMP_NUM_THREADS=", thread))
      out_report_name <- paste("traces/CPU", thread, n_images, do_it_again, sep="_")
      cmd <- paste0(nsys, nsys_args, out_report_name, " ",
                    paste(cpu_bin, images, n_images, sep=" "))
      print(cmd)
      system(flush)
      system(cmd)
  }
}

#for(do_it_again in seq(1, 10, 1)){
for(do_it_again in seq(1, 1, 1)){    
    for(batch in c(1, 500, 1000, 1200)){
        for(stream in c(2 ** seq(0, 4))){
        out_report_name <- paste("traces/GPU", batch, n_images, stream, do_it_again, sep="_")
        cmd <- paste0(nsys, nsys_args, out_report_name, " ",
                      paste(gpu_bin, images, batch, n_images, stream, sep=" "))
        print(cmd)
        system(flush)
        system(cmd)
        }
    }        
}
