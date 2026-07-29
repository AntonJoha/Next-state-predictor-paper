

_:
	python -m next_state_predictor.main 

test_acrobot:
	python -m next_state_predictor.main --env Acrobot-v1  --num_episodes 10 


commit: lint test test_acrobot 
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
