import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

def analyze_jira_data(file_path, target_date=None):
    """
    Analyze Jira project data to predict completion date and visualize project status.
    
    Parameters:
    file_path (str): Path to the Jira export CSV file
    target_date (str): Optional target date in format 'YYYY-MM-DD'
    
    Returns:
    dict: Analysis results including predicted completion date and risk assessment
    """
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    # Make all column names lowercase for consistent referencing
    df.columns = df.columns.str.lower()
    
    # Print columns for debugging
    print("Available columns:", df.columns.tolist())
    
    # Convert relevant date columns to datetime
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'created' in col.lower() or 'updated' in col.lower()]
    print(f"Detected date columns: {date_cols}")
    
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Ensure story points column exists, with common variations in naming
    story_point_cols = [col for col in df.columns if 'story' in col.lower() and 'point' in col.lower() or 'sp' == col.lower()]
    if story_point_cols:
        story_point_col = story_point_cols[0]
        print(f"Using '{story_point_col}' for story points")
    else:
        # If no story point column found, create a default one
        df['story points'] = np.nan
        story_point_col = 'story points'
        print("No story points column found, created default 'story points' column")
    
    # Ensure status column exists
    status_cols = [col for col in df.columns if 'status' in col.lower()]
    if status_cols:
        status_col = status_cols[0]
        print(f"Using '{status_col}' for ticket status")
    else:
        print("Warning: No status column found.")
        status_col = None
    
    # Basic project statistics
    total_tickets = len(df)
    print(f"Total tickets in dataset: {total_tickets}")
    
    # Filter to only include tickets with story points
    estimated_tickets = df[~df[story_point_col].isna()]
    unestimated_tickets = df[df[story_point_col].isna()]
    print(f"Tickets with story points: {len(estimated_tickets)}")
    print(f"Tickets without story points: {len(unestimated_tickets)}")
    
    # Completion Analysis
    if status_col:
        # Determine completed/incomplete tickets based on status
        # Assuming 'Done', 'Closed', 'Resolved' indicate completion - adjust as needed
        completed_statuses = ['Done', 'Closed', 'Resolved', 'Complete']
        df['is_completed'] = df[status_col].str.lower().isin([s.lower() for s in completed_statuses])
        
        completed_tickets = df[df['is_completed']]
        incomplete_tickets = df[~df['is_completed']]
        
        print(f"Completed tickets: {len(completed_tickets)}")
        print(f"Incomplete tickets: {len(incomplete_tickets)}")
        
        completed_points = completed_tickets[story_point_col].sum()
        remaining_estimated_points = incomplete_tickets[~incomplete_tickets[story_point_col].isna()][story_point_col].sum()
        
        # Handle tickets without story points by estimating
        if len(unestimated_tickets) > 0 and len(estimated_tickets) > 0:
            # Calculate average story points per estimated ticket
            avg_story_points = estimated_tickets[story_point_col].mean()
            
            # Estimate points for unestimated incomplete tickets
            num_unestimated_incomplete = len(incomplete_tickets[incomplete_tickets[story_point_col].isna()])
            estimated_remaining_unestimated_points = num_unestimated_incomplete * avg_story_points
            
            total_remaining_points = remaining_estimated_points + estimated_remaining_unestimated_points
        else:
            total_remaining_points = remaining_estimated_points
            estimated_remaining_unestimated_points = 0
            avg_story_points = 0
    else:
        print("Warning: Cannot determine completion status without a status column.")
        completed_tickets = pd.DataFrame()
        incomplete_tickets = df
        completed_points = 0
        remaining_estimated_points = df[~df[story_point_col].isna()][story_point_col].sum()
        avg_story_points = estimated_tickets[story_point_col].mean() if len(estimated_tickets) > 0 else 0
        num_unestimated_incomplete = len(incomplete_tickets[incomplete_tickets[story_point_col].isna()])
        estimated_remaining_unestimated_points = num_unestimated_incomplete * avg_story_points
        total_remaining_points = remaining_estimated_points + estimated_remaining_unestimated_points
    
    # Velocity Calculation
    if len(completed_tickets) > 0 and 'created' in df.columns:
        # Find the earliest and latest completion dates
        if 'resolution date' in df.columns:
            earliest_activity = df['created'].min()
            latest_completion = completed_tickets['resolution date'].max()
            print(f"Earliest activity: {earliest_activity}")
            print(f"Latest completion: {latest_completion}")
        else:
            # If no resolution date, use the latest updated date
            updated_cols = [col for col in df.columns if 'updated' in col.lower()]
            if updated_cols:
                earliest_activity = df['created'].min()
                latest_completion = completed_tickets[updated_cols[0]].max()
                print(f"Earliest activity: {earliest_activity}")
                print(f"Latest completion (using updated date): {latest_completion}")
            else:
                # Fallback to using created date with an offset
                earliest_activity = df['created'].min()
                latest_completion = df['created'].max()
                print(f"Earliest activity: {earliest_activity}")
                print(f"Latest completion (fallback to created max): {latest_completion}")
        
        if pd.notna(earliest_activity) and pd.notna(latest_completion):
            # Calculate project duration in days
            project_duration = (latest_completion - earliest_activity).days
            if project_duration > 0:
                # Calculate average velocity in points per day
                velocity_per_day = completed_points / project_duration
                
                # Weekly velocity
                velocity_per_week = velocity_per_day * 7
                
                print(f"Project duration: {project_duration} days")
                print(f"Velocity: {velocity_per_day:.2f} points/day ({velocity_per_week:.2f} points/week)")
            else:
                velocity_per_day = 0
                velocity_per_week = 0
                print("Warning: Project duration is zero or negative, setting velocity to 0")
        else:
            velocity_per_day = 0
            velocity_per_week = 0
            print("Warning: Invalid date values for earliest activity or latest completion")
    else:
        # If no completed tickets or dates available, assume a default velocity
        velocity_per_day = 1  # Placeholder
        velocity_per_week = 7
        project_duration = 0
        print("Warning: Using default velocity (1 point/day) due to missing completed tickets or dates")
    
    # Prediction of completion date
    if velocity_per_day > 0:
        days_to_completion = total_remaining_points / velocity_per_day
        predicted_completion_date = datetime.now() + timedelta(days=days_to_completion)
        print(f"Predicted completion in {days_to_completion:.1f} days ({predicted_completion_date.strftime('%Y-%m-%d')})")
        
        # Risk assessment
        if target_date:
            target_dt = pd.to_datetime(target_date)
            days_until_target = (target_dt - datetime.now()).days
            
            if days_until_target >= days_to_completion * 1.2:
                risk_level = "Low"
                risk_explanation = f"You're likely to complete {days_until_target - days_to_completion:.1f} days ahead of schedule."
            elif days_until_target >= days_to_completion * 0.9:
                risk_level = "Medium"
                risk_explanation = "You're cutting it close. Consider adding buffer time."
            else:
                risk_level = "High"
                risk_explanation = f"You're likely to miss the target by approximately {days_to_completion - days_until_target:.1f} days."
            print(f"Risk assessment for target date {target_date}: {risk_level}")
            print(risk_explanation)
        else:
            risk_level = "Unknown"
            risk_explanation = "No target date provided for risk assessment."
    else:
        predicted_completion_date = "Unable to predict (insufficient velocity data)"
        risk_level = "Unknown"
        risk_explanation = "Cannot calculate risk without velocity data."
        print("Warning: Unable to predict completion date due to zero velocity")
    
    # Create visualizations directory
    output_dir = "jira_analysis_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")
    
    # Generate Visualizations
    # 1. Status Distribution
    if status_col:
        plt.figure(figsize=(10, 6))
        status_counts = df[status_col].value_counts()
        sns.barplot(x=status_counts.index, y=status_counts.values)
        plt.title("Ticket Status Distribution")
        plt.xlabel("Status")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/status_distribution.png")
        plt.close()
        print("Generated status distribution chart")
    
    # 2. Story Points Distribution
    plt.figure(figsize=(10, 6))
    df[~df[story_point_col].isna()][story_point_col].hist(bins=10)
    plt.title("Story Points Distribution")
    plt.xlabel("Story Points")
    plt.ylabel("Number of Tickets")
    plt.savefig(f"{output_dir}/story_points_distribution.png")
    plt.close()
    print("Generated story points distribution chart")
    
    # 3. Burndown Chart
    print("Checking burndown chart prerequisites...")
    print(f"Created column in date_cols: {'created' in df.columns}")
    print(f"Completed tickets: {len(completed_tickets)}")
    
    if 'created' in df.columns and len(completed_tickets) > 0:
        print("Prerequisites met for burndown chart")
        plt.figure(figsize=(12, 6))
        
        # Sort all tickets by creation date
        sorted_df = df.sort_values('created')
        
        # Calculate cumulative story points
        sorted_df[story_point_col] = sorted_df[story_point_col].fillna(avg_story_points)
        cumulative_total = sorted_df[story_point_col].cumsum()
        
        # For completed tickets, calculate when points were completed
        if 'resolution date' in df.columns:
            print("Using resolution date for burndown chart")
            completed_sorted = completed_tickets.sort_values('resolution date')
            dates = completed_sorted['resolution date'].dropna()
            completed_sorted[story_point_col] = completed_sorted[story_point_col].fillna(avg_story_points)
            completed_cumulative = completed_sorted[story_point_col].cumsum()
        else:
            # Fallback using created date
            print("Fallback to using updated date for burndown chart")
            if 'updated' in df.columns:
                completed_sorted = completed_tickets.sort_values('updated')
                dates = completed_sorted['updated'].dropna()
            else:
                completed_sorted = completed_tickets.sort_values('created')
                dates = completed_sorted['created'].dropna()
            
            completed_sorted[story_point_col] = completed_sorted[story_point_col].fillna(avg_story_points)
            completed_cumulative = completed_sorted[story_point_col].cumsum()
        
        # Plot total scope
        plt.plot(sorted_df['created'], cumulative_total, label='Total Scope')
        print(f"Plotted total scope line with {len(sorted_df)} points")
        
        # Plot completed work
        if len(dates) > 0:
            plt.plot(dates, completed_cumulative, label='Completed Work')
            print(f"Plotted completed work line with {len(dates)} points")
        else:
            print("Warning: No valid dates for completed work")
        
        # Plot ideal burndown if target date is provided
        if target_date:
            target_dt = pd.to_datetime(target_date)
            start_date = sorted_df['created'].min()
            total_points = cumulative_total.iloc[-1]
            
            # Generate ideal burndown line
            ideal_dates = [start_date, target_dt]
            ideal_points = [0, total_points]
            plt.plot(ideal_dates, ideal_points, 'r--', label='Ideal Burndown')
            print(f"Plotted ideal burndown line from {start_date} to {target_dt}")
            
            # Add predicted completion
            if isinstance(predicted_completion_date, datetime):
                plt.axvline(x=predicted_completion_date, color='orange', linestyle='--', 
                           label=f'Predicted: {predicted_completion_date.strftime("%Y-%m-%d")}')
                print(f"Added predicted completion line at {predicted_completion_date}")
            
            plt.axvline(x=target_dt, color='green', linestyle='--', 
                       label=f'Target: {target_dt.strftime("%Y-%m-%d")}')
            print(f"Added target date line at {target_dt}")
        
        plt.title("Project Burndown Chart")
        plt.xlabel("Date")
        plt.ylabel("Story Points")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        try:
            plt.savefig(f"{output_dir}/burndown_chart.png")
            print("Successfully saved burndown chart")
        except Exception as e:
            print(f"Error saving burndown chart: {e}")
        
        plt.close()
    else:
        print("Skipping burndown chart - missing prerequisites")
        if 'created' not in df.columns:
            print("  - 'created' column not found")
        if len(completed_tickets) == 0:
            print("  - No completed tickets found")
    
    # 4. Velocity Chart
    if len(completed_tickets) > 0 and 'created' in df.columns:
        # If we have resolution dates, use those for calculating weekly velocity
        try:
            if 'resolution date' in df.columns:
                completed_tickets['week'] = completed_tickets['resolution date'].dt.isocalendar().week
                completed_tickets['year'] = completed_tickets['resolution date'].dt.isocalendar().year
                print("Using resolution date for velocity chart")
            else:
                # Fallback to created date
                completed_tickets['week'] = completed_tickets['created'].dt.isocalendar().week
                completed_tickets['year'] = completed_tickets['created'].dt.isocalendar().year
                print("Using created date for velocity chart (fallback)")
            
            # Group by week and sum story points
            weekly_velocity = completed_tickets.groupby(['year', 'week'])[story_point_col].sum().reset_index()
            weekly_velocity['week_label'] = weekly_velocity['year'].astype(str) + '-W' + weekly_velocity['week'].astype(str)
            
            plt.figure(figsize=(12, 6))
            sns.barplot(x='week_label', y=story_point_col, data=weekly_velocity)
            plt.title("Weekly Velocity")
            plt.xlabel("Week")
            plt.ylabel("Story Points Completed")
            plt.xticks(rotation=45)
            plt.axhline(y=velocity_per_week, color='r', linestyle='--', label=f'Avg: {velocity_per_week:.1f} pts/week')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/velocity_chart.png")
            plt.close()
            print("Generated velocity chart")
        except Exception as e:
            print(f"Error generating velocity chart: {e}")
    
    # 5. Estimated vs. Unestimated Tickets Pie Chart
    plt.figure(figsize=(8, 8))
    estimated_count = len(estimated_tickets)
    unestimated_count = len(unestimated_tickets)
    plt.pie([estimated_count, unestimated_count], 
            labels=['Estimated', 'Unestimated'], 
            autopct='%1.1f%%', 
            colors=['#66b3ff', '#ffcc99'])
    plt.title("Estimated vs. Unestimated Tickets")
    plt.savefig(f"{output_dir}/estimation_coverage.png")
    plt.close()
    print("Generated estimation coverage chart")
    
    # Compile analysis results
    results = {
        "total_tickets": total_tickets,
        "estimated_tickets": len(estimated_tickets),
        "unestimated_tickets": len(unestimated_tickets),
        "completed_tickets": len(completed_tickets) if 'is_completed' in df.columns else 'Unknown',
        "incomplete_tickets": len(incomplete_tickets) if 'is_completed' in df.columns else 'Unknown',
        "avg_story_points": avg_story_points,
        "completed_points": completed_points,
        "remaining_estimated_points": remaining_estimated_points,
        "estimated_remaining_unestimated_points": estimated_remaining_unestimated_points,
        "total_remaining_points": total_remaining_points,
        "velocity_per_day": velocity_per_day,
        "velocity_per_week": velocity_per_week,
        "predicted_completion_date": predicted_completion_date,
        "risk_level": risk_level,
        "risk_explanation": risk_explanation,
        "output_directory": output_dir
    }
    
    # Print summary
    print("\n===== JIRA PROJECT ANALYSIS SUMMARY =====")
    print(f"Total tickets: {total_tickets}")
    print(f"Completed tickets: {len(completed_tickets) if 'is_completed' in df.columns else 'Unknown'}")
    print(f"Incomplete tickets: {len(incomplete_tickets) if 'is_completed' in df.columns else 'Unknown'}")
    print(f"Tickets with story points: {len(estimated_tickets)}")
    print(f"Tickets without story points: {len(unestimated_tickets)}")
    print(f"\nAverage story points per ticket: {avg_story_points:.2f}")
    print(f"Completed story points: {completed_points}")
    print(f"Remaining estimated story points: {remaining_estimated_points}")
    print(f"Estimated points for unestimated tickets: {estimated_remaining_unestimated_points:.2f}")
    print(f"Total remaining points: {total_remaining_points:.2f}")
    print(f"\nCurrent velocity: {velocity_per_day:.2f} points/day ({velocity_per_week:.2f} points/week)")
    print(f"Predicted completion date: {predicted_completion_date if isinstance(predicted_completion_date, datetime) else predicted_completion_date}")
    
    if target_date:
        print(f"\nTarget date: {target_date}")
        print(f"Risk assessment: {risk_level}")
        print(f"Risk explanation: {risk_explanation}")
    
    print(f"\nVisualization outputs saved to: {output_dir}/")
    
    return results

def main():
    """
    Main function to execute when run as a script.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze Jira project data and predict completion.')
    parser.add_argument('file_path', help='Path to the Jira export CSV file')
    parser.add_argument('--target', help='Optional target date in format YYYY-MM-DD')
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_jira_data(args.file_path, args.target)

if __name__ == "__main__":
    main()