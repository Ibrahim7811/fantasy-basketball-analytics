# Fantasy Basketball Analytics — Head-to-Head Matchup Simulator

#### Video Demo: [https://youtu.be/defn-YKVqFQ]

#### Description:

### Project Overview

The main purpose of this project is to find the highest projected fantasy point totals for different fantasy basketball team lineups. The project also simulates a head-to-head matchup between two fantasy teams over a specific date range by comparing their projected points. This project is especially useful for people who want to know how their drafted fantasy basketball team might perform against another team. It also provides the user with the ability to analyze a matchup against an opponent over a given date range, usually during a week of the regular NBA season. The projections and game schedules used by the program are provided by the user through CSV files.

### Features

The project incorporates features that determine whether certain players on a fantasy team are eligible for a given roster slot and whether a player is scheduled to play a game on a particular day within the date range. These two features are used by the greedy lineup algorithm and the optimal lineup algorithm, which both generate fantasy basketball lineups based on the players' projected fantasy points. The greedy algorithm selects players based on their projections and eligibility, while the optimal algorithm also considers how each assignment affects the remaining available lineup slots.

These lineup algorithms, combined with the provided date range, make weekly lineup optimization possible by generating an optimized lineup for every day of the matchup. Another feature that users can take advantage of is the ability to customize the starting lineup slots according to their own fantasy basketball league. The program also loads player information, eligible positions, projected points, schedules, and NBA teams from user-provided CSV files. The CSV data is validated to prevent malformed files or unsupported inputs from being used. A command-line interface makes the program easier to use by allowing users to provide their files, date ranges, and lineup configurations, while displaying clear error messages when an input is invalid.

All of these features come together in the head-to-head matchup analysis. This feature runs the weekly lineup optimizer separately for both fantasy teams, calculates their projected weekly totals, finds the point differential, and determines the projected winner. The results are then organized into a readable report containing the daily projections, weekly totals, and projected winner.

### How It Works

The first lineup algorithm was implemented using a greedy approach, which sorts players by their projected fantasy points and assigns the highest projected eligible players to the available lineup slots. Although this algorithm checks whether a player is eligible for a slot, it does not fully consider how assigning that player might affect the remaining slots and players. This means that it does not always produce the highest possible projected lineup total.

This is where the optimal lineup algorithm comes into play. Unlike the greedy approach, the optimal algorithm uses recursive backtracking to explore different eligible player assignments. It first takes the list of players and filters out those who do not have a game on the selected date. It then considers positional eligibility, sorts the available players by their projected points, and begins building different lineup combinations. The algorithm makes sure that no player is used more than once in the same daily lineup and also allows slots to remain empty when they cannot be filled.

The backtracking algorithm also implements a branch-and-bound technique to avoid unnecessary searches. During the search, the program looks at how many lineup slots remain and estimates the highest additional projected points that could be gained from the best remaining players. Those projected points are added to the current lineup's score to calculate an optimistic upper bound. If this upper bound is not greater than the best lineup score already found, the algorithm abandons that branch because it cannot improve the result. This reduces the number of recursive calls needed while still allowing the algorithm to find the highest-scoring eligible lineup.

After a daily lineup has been optimized, the weekly optimizer repeats the process for every date in the selected range and adds the daily projected totals together. The matchup analyzer performs this weekly calculation for both teams, compares their projections, and passes the results to the reporting function, which formats the information into a table.

### Project File Structure

The project is separated into several Python files, with each file handling a specific part of the program:

- **`project.py`** is the main program. It reads the command-line arguments, validates the selected dates and lineup slots, loads both fantasy teams, and displays the final matchup report.
- **`player_class.py`** contains the `Player` class, which stores a player's name, team, eligible positions, projected points, and scheduled games. It also contains methods that determine whether a player has a game on a selected date and is eligible for a particular lineup slot.
- **`data_loader.py`** reads the user-provided CSV files, creates the `Player` objects, and checks that the player information is formatted correctly.
- **`lineup.py`** contains the greedy and optimal lineup algorithms, along with the function that calculates the total projected points of a lineup.
- **`weekly.py`** takes a start date and end date, generates the dates within that range, and optimizes a lineup for each day to calculate the weekly projection.
- **`matchup.py`** analyzes the two fantasy teams by running the weekly optimizer for both, finding the point differential, and determining the projected winner.
- **`report.py`** uses the `tabulate` library to format the daily and weekly projections into a readable head-to-head matchup report.
- **`integration.py`** and **`benchmark_lineup.py`** were used during development to connect the earlier components and measure the performance of the lineup optimizer.
- **`test_project.py`** and the other **`test_*.py`** files contain automated tests for the different functions and modules.
- **`requirements.txt`** lists the external Python packages needed for the project.

### Installation and Requirements

The project was developed and tested using Python 3.14.7. Before running the application, the user should have Python installed and open a terminal in the project folder. The external libraries used by the program are `tabulate`, which formats the matchup report, and `pytest`, which runs the automated tests.

The required libraries can be installed with the following command:

```bash
python3 -m pip install -r requirements.txt
```

The `requirements.txt` file contains the following packages:

```text
tabulate==0.10.0
pytest==9.1.1
```

### Usage and CSV Format

To use the program, the user needs to provide two CSV files, one containing the players on Team A and another containing the players on Team B. Each file must contain the following columns: `name`, `positions`, `projected_points`, `schedule`, and `team`.

An example of the CSV format is shown below:

```csv
name,positions,projected_points,schedule,team
Nikola Jokic,C,58.5,2026-11-09;2026-11-11,DEN
Cade Cunningham,PG;SG,48.25,2026-11-09;2026-11-12,DET
```

Each row represents one player. If a player is eligible for multiple positions or has multiple scheduled games, those values are separated using semicolons. The projected points represent the player's expected fantasy points for a game, and the schedule contains the dates on which the player is expected to play. These values are supplied by the user.

Once both CSV files are prepared, the program can be run from the terminal using the following command:

```bash
python3 project.py \
  --team-a players.csv \
  --team-b opponent.csv \
  --start 2026-11-09 \
  --end 2026-11-15
```

The `--team-a` and `--team-b` arguments specify the paths to the two CSV files, while `--start` and `--end` specify the beginning and ending dates of the matchup. The program will then analyze every day within the provided date range, including both the start and end dates.

Users can also customize the available lineup slots using the optional `--slots` argument. For example:

```bash
python3 project.py \
  --team-a players.csv \
  --team-b opponent.csv \
  --start 2026-11-09 \
  --end 2026-11-15 \
  --slots PG,SG,C1
```

If the `--slots` argument is not provided, the program uses the default configuration of `PG`, `SG`, `G`, `SF`, `PF`, `F`, `C1`, `C2`, `UTIL1`, and `UTIL2`. After the matchup is analyzed, the final report displays the projected points for both teams on each date, their weekly totals, the projected winner, and the point differential. Users can also run `python3 project.py --help` to see the available command-line arguments.

### Testing

The project includes automated tests using `pytest` to verify that its different functions work as expected. The tests cover player schedules and positional eligibility, CSV data loading and validation, greedy and optimal lineup selection, weekly lineup calculations, head-to-head matchup results, report formatting, and command-line input handling. They also check cases such as players without scheduled games, invalid projections, missing CSV columns, duplicate lineup slots, and incorrectly ordered dates.

The entire test suite can be run from the project folder with:

```bash
python3 -m pytest -v
```

Individual test files can also be run separately, such as `python3 -m pytest test_project.py -v`. These tests were used throughout the development of the project to make sure that new features continued to work with the existing parts of the program.
