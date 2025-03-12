# Jira Project Analysis Tool made with Claude.ai

```
python jira_analysis.py path/to/jira_export.csv --target 2025-04-30
```


I'll explain each of the visualizations that the script generates to help you understand your Jira project data:
1. Status Distribution
This bar chart shows how many tickets are in each status category (To Do, In Progress, In Review, Done, etc.). It gives you an immediate visual of work distribution across your project.
What to look for:

Large number of tickets in "To Do" may indicate potential bottlenecks ahead
Many tickets in "In Progress" simultaneously could signal context switching or work overload
Balance between different stages helps identify workflow issues

2. Story Points Distribution
This histogram shows the frequency of different story point values across all estimated tickets.
What to look for:

The most common story point values used by your team
Whether your team is using a wide or narrow range of story point values
Unusually large stories that might need breaking down
Consistency in estimation practices

3. Burndown Chart
This is perhaps the most important visualization for tracking progress and predicting completion. It shows:

Total Scope (blue line): The cumulative story points over time
Completed Work (orange line): Story points completed over time
Ideal Burndown (red dashed line): The perfect pace to meet your target date
Predicted Completion (orange vertical line): When you're likely to finish based on current velocity
Target Date (green vertical line): Your specified deadline

What to look for:

Gap between "ideal" and "actual" burn lines indicates if you're ahead or behind schedule
Flat periods in the completed work line suggest slowdowns or blockers
Whether the predicted completion falls before or after your target date
Scope changes (jumps in the total scope line)

4. Velocity Chart
This bar chart shows how many story points your team completed each week, with a red dashed line indicating your average weekly velocity.
What to look for:

Consistency or volatility in your team's output
Upward or downward trends in velocity
Weeks with unusually high or low productivity (may warrant discussion)
How recent velocity compares to historical average

5. Estimated vs. Unestimated Tickets Pie Chart
This pie chart visualizes what percentage of your tickets have story point estimates versus those that don't.
What to look for:

High percentage of unestimated tickets suggests estimation gaps
This is important because unestimated tickets introduce uncertainty into predictions
Your goal should typically be to increase the "Estimated" portion over time

Together, these visualizations provide a comprehensive view of:

Current project status
Work distribution
Historical performance
Prediction reliability
Potential risks to meeting your deadline

The script combines these visuals with numerical calculations to give you both intuitive and concrete data for making project decisions.