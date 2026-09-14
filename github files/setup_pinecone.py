#!/usr/bin/env python3
"""
Pinecone Setup Script
Helps set up Pinecone configuration and upload embeddings.
"""

import os
import shutil
from pathlib import Path

def setup_pinecone():
    """Set up Pinecone configuration."""
    print("Pinecone Setup for Liver Cirrhosis Research Papers")
    print("=" * 50)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file already exists")
        response = input("Do you want to recreate it? (y/n): ").strip().lower()
        if response != 'y':
            print("Using existing .env file")
            return
        else:
            # Backup existing .env
            shutil.copy('.env', '.env.backup')
            print("Backed up existing .env to .env.backup")
    
    print("\nSetting up Pinecone configuration...")
    print("You'll need your Pinecone API key from: https://app.pinecone.io/organizations")
    
    # Get API key
    api_key = input("\nEnter your Pinecone API key: ").strip()
    if not api_key:
        print("❌ API key is required")
        return
    
    # Get environment (optional)
    print("\nPinecone Environment (press Enter for default 'us-east-1-aws'):")
    environment = input("Environment: ").strip() or "us-east-1-aws"
    
    # Get index name
    print("\nIndex name (press Enter for default 'liver-cirrhosis-embeddings'):")
    index_name = input("Index name: ").strip() or "liver-cirrhosis-embeddings"
    
    # Create .env file
    env_content = f"""# Pinecone Configuration
PINECONE_API_KEY={api_key}
PINECONE_ENVIRONMENT={environment}
INDEX_NAME={index_name}
DIMENSION=384
METRIC=cosine
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ .env file created successfully!")
    
    # Check if embeddings exist
    embeddings_dir = Path("embeddings")
    if not embeddings_dir.exists():
        print("❌ Embeddings directory not found!")
        print("Please run embedding_generator.py first to create embeddings.")
        return
    
    # Check if embeddings.npy exists
    embeddings_file = embeddings_dir / "embeddings.npy"
    if not embeddings_file.exists():
        print("❌ Embeddings file not found!")
        print("Please run embedding_generator.py first to create embeddings.")
        return
    
    print("✅ Embeddings found!")
    
    # Ask if user wants to upload to Pinecone
    response = input("\nDo you want to upload embeddings to Pinecone now? (y/n): ").strip().lower()
    if response == 'y':
        print("\nUploading embeddings to Pinecone...")
        try:
            from pinecone_manager import PineconeManager
            manager = PineconeManager()
            manager.process_all_embeddings()
            manager.print_summary()
            print("✅ Embeddings uploaded successfully!")
        except Exception as e:
            print(f"❌ Failed to upload embeddings: {e}")
            print("You can try running: python pinecone_manager.py")
    
    print("\nSetup complete!")
    print("\nNext steps:")
    print("1. Run: python pinecone_search.py (for interactive search)")
    print("2. Or use the PineconeManager class in your own scripts")

def main():
    """Main function."""
    try:
        setup_pinecone()
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
    except Exception as e:
        print(f"❌ Error during setup: {e}")

if __name__ == "__main__":
    main()

