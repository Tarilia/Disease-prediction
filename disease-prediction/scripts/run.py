#!/usr/bin/env python


from disease_prediction.launch import to_run


def main():
    print("Let's analyze the dataset and predict early-stage Parkinson's"
          " disease using XGBoost machine learning algorithm and sklearn"
          " library for feature normalization. Visualize the results"
          " and generate a report.")
    to_run()


if __name__ == '__main__':
    main()
