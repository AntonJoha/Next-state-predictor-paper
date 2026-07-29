

_:
	python -m next_state_predictor.main 


commit: lint clean test
	git add .
	git commit -m "commit"
	git push


test:
	python -m next_state_predictor.main 

clean: 
	rm -rf results_dev
hardclean: 
	rm -rf results_dev
	rm -rf results

lint:
	./lint.sh
