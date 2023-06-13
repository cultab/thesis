render:
	R --quiet -e "require(rmarkdown); render('report.rmd', output_format='all');"

submit:
	cp report.pdf 171014.pdf
	7z a 171014.zip 171014.pdf

.PHONY: render submit
