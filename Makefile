

_:
	python -m next_state_predictor.main 



commit: lint test test_acrobot test_mlp
	git add .
	git commit -m "commit"
	git push



test_mlp:
	python -m next_state_predictor.main  --num_episodes 10 --next_state_predictor mlp

test_acrobot:
	python -m next_state_predictor.main --env Acrobot-v1  --num_episodes 10 


test:
	python -m next_state_predictor.main 

clean: 
	rm -rf results_dev
hardclean: 
	rm -rf results_dev
	rm -rf results

lint:
	./lint.sh
