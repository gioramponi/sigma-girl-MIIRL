import pandas as pd


if __name__ == "__main__":
    # ! currently generates ranking for each trans, do for traj or not needed?
    # ! why needed to stand/norm before ranking, ranking is unchanged
    data_file = "highway-env/condensed_binary_highway_data.csv"
    df = pd.read_csv(data_file)

    # Calculate the total number of collisions and offroad visits for each trajectory
    df['total_collisions'] = df.filter(regex='num_collisions').sum(axis=1)
    df['total_offroad_visits'] = df.filter(regex='num_offroad_visits').sum(axis=1)

    # Create a safety score for each trajectory by adding the number of collisions and offroad visits
    df['safety_score'] = df['num_collisions'] + df['num_offroad_visits']

    # Sort the trajectories based on their safety score in descending order
    df = df.sort_values(by='safety_score', ascending=False)

    # df = df[df['step'] == 39]
    # df["safety_score"] = df[["num_collisions", "num_offroad_visits"]]\
    #     .apply(tuple,axis=1).rank(method='dense',ascending=False).astype(int)

    # for data_type in ["", "_normalized", "_standradized"]:
    #     num_collisions = "num_collisions" + data_type
    #     num_offroad_visits = "num_offroad_visits" + data_type

    #     df["safety_score" + data_type] = df[[num_collisions, num_offroad_visits]]\
    #         .apply(tuple,axis=1).rank(method='dense',ascending=False).astype(int)

    df.to_csv("data_safety_rankings.csv")