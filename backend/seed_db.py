#!/usr/bin/env python3
"""
Database seeding script
Run this to populate the database with sample data
"""

import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.seed_questions import seed_sample_questions


async def main():
    """Main seeding function"""
    print("🌱 Starting database seeding...")
    
    try:
        await seed_sample_questions()
        print("✅ Database seeding completed successfully!")
    except Exception as e:
        print(f"❌ Database seeding failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())