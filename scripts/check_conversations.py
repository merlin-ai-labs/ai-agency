"""Script to check if conversations and tool calls are saved in the database.

Checks:
1. LangGraph checkpoints table (langgraph_checkpoints)
2. Conversations table (conversations)
3. Messages table (messages)
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlmodel import Session, select, text
from app.db.base import get_session
from app.db.models import Conversation, Message
from app.config import settings


def check_langgraph_checkpoints(conversation_id: str | None = None, tenant_id: str | None = None):
    """Check LangGraph checkpoints."""
    print("=" * 70)
    print("LangGraph Checkpoints")
    print("=" * 70)
    
    with Session(get_session()) as session:
        # Check if table exists
        try:
            result = session.exec(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'langgraph_checkpoints'
                )
            """))
            table_exists = result.first()
            
            if not table_exists:
                print("❌ langgraph_checkpoints table does not exist")
                print("   Run migrations: alembic upgrade head")
                return
            
            # Query checkpoints
            query = "SELECT thread_id, checkpoint_ns, checkpoint, checkpoint_data, parent_checkpoint_id, created_at FROM langgraph_checkpoints"
            conditions = []
            
            if conversation_id:
                conditions.append(f"thread_id = '{conversation_id}'")
            if tenant_id:
                # Check tenant_id in checkpoint_data JSON
                conditions.append(f"checkpoint_data::text LIKE '%{tenant_id}%'")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT 10"
            
            result = session.exec(text(query))
            checkpoints = result.fetchall()
            
            if not checkpoints:
                print("⚠️  No checkpoints found")
                if conversation_id:
                    print(f"   Searched for conversation_id: {conversation_id}")
                if tenant_id:
                    print(f"   Searched for tenant_id: {tenant_id}")
            else:
                print(f"✅ Found {len(checkpoints)} checkpoint(s):\n")
                for i, cp in enumerate(checkpoints, 1):
                    print(f"Checkpoint {i}:")
                    print(f"  Thread ID: {cp[0]}")
                    print(f"  Namespace: {cp[1]}")
                    print(f"  Parent: {cp[4]}")
                    print(f"  Created: {cp[5]}")
                    if cp[3]:  # checkpoint_data
                        print(f"  Data: {str(cp[3])[:100]}...")
                    print()
        
        except Exception as e:
            print(f"❌ Error querying checkpoints: {e}")


def check_conversations(conversation_id: str | None = None, tenant_id: str | None = None, flow_type: str = "invoice_manager"):
    """Check conversations table."""
    print("=" * 70)
    print("Conversations Table")
    print("=" * 70)
    
    with Session(get_session()) as session:
        query = select(Conversation)
        
        if conversation_id:
            query = query.where(Conversation.conversation_id == conversation_id)
        if tenant_id:
            query = query.where(Conversation.tenant_id == tenant_id)
        if flow_type:
            query = query.where(Conversation.flow_type == flow_type)
        
        conversations = session.exec(query.order_by(Conversation.created_at.desc()).limit(10)).all()
        
        if not conversations:
            print("⚠️  No conversations found")
            if conversation_id:
                print(f"   Searched for conversation_id: {conversation_id}")
            if tenant_id:
                print(f"   Searched for tenant_id: {tenant_id}")
            print(f"   Flow type: {flow_type}")
            print("\n   Note: Invoice manager currently uses LangGraph checkpoints,")
            print("   not the conversations table. Consider adding conversation persistence.")
        else:
            print(f"✅ Found {len(conversations)} conversation(s):\n")
            for conv in conversations:
                print(f"Conversation ID: {conv.conversation_id}")
                print(f"  Tenant ID: {conv.tenant_id}")
                print(f"  Flow Type: {conv.flow_type}")
                print(f"  Created: {conv.created_at}")
                print(f"  Updated: {conv.updated_at}")
                if conv.flow_metadata:
                    print(f"  Metadata: {conv.flow_metadata}")
                print()


def check_messages(conversation_id: str | None = None, tenant_id: str | None = None, flow_type: str = "invoice_manager"):
    """Check messages table."""
    print("=" * 70)
    print("Messages Table")
    print("=" * 70)
    
    with Session(get_session()) as session:
        query = select(Message)
        
        if conversation_id:
            query = query.where(Message.conversation_id == conversation_id)
        if tenant_id:
            query = query.where(Message.tenant_id == tenant_id)
        if flow_type:
            query = query.where(Message.flow_type == flow_type)
        
        messages = session.exec(query.order_by(Message.created_at.desc()).limit(20)).all()
        
        if not messages:
            print("⚠️  No messages found")
            if conversation_id:
                print(f"   Searched for conversation_id: {conversation_id}")
            if tenant_id:
                print(f"   Searched for tenant_id: {tenant_id}")
            print(f"   Flow type: {flow_type}")
            print("\n   Note: Invoice manager currently uses LangGraph checkpoints,")
            print("   not the messages table. Consider adding message persistence.")
        else:
            print(f"✅ Found {len(messages)} message(s):\n")
            for msg in messages:
                print(f"Message ID: {msg.id}")
                print(f"  Conversation ID: {msg.conversation_id}")
                print(f"  Role: {msg.role}")
                print(f"  Content: {msg.content[:100]}..." if len(msg.content) > 100 else f"  Content: {msg.content}")
                if msg.tool_calls:
                    print(f"  Tool Calls: {msg.tool_calls}")
                if msg.message_metadata:
                    print(f"  Metadata: {msg.message_metadata}")
                print(f"  Created: {msg.created_at}")
                print()


def check_tool_executions():
    """Check for tool execution records."""
    print("=" * 70)
    print("Tool Executions")
    print("=" * 70)
    
    with Session(get_session()) as session:
        # Check messages with tool_calls
        query = select(Message).where(Message.tool_calls.isnot(None))
        messages_with_tools = session.exec(query.order_by(Message.created_at.desc()).limit(10)).all()
        
        if not messages_with_tools:
            print("⚠️  No messages with tool calls found")
            print("   Tool calls are stored in messages.tool_calls JSON field")
        else:
            print(f"✅ Found {len(messages_with_tools)} message(s) with tool calls:\n")
            for msg in messages_with_tools:
                print(f"Conversation: {msg.conversation_id}")
                print(f"  Role: {msg.role}")
                print(f"  Tool Calls: {msg.tool_calls}")
                print(f"  Created: {msg.created_at}")
                print()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Check database for saved conversations and tool calls")
    parser.add_argument("--conversation-id", help="Specific conversation ID to check")
    parser.add_argument("--tenant-id", default="test_tenant", help="Tenant ID to check")
    parser.add_argument("--flow-type", default="invoice_manager", help="Flow type to check")
    parser.add_argument("--all", action="store_true", help="Check all tables")
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("Database Check - Conversations & Tool Calls")
    print("=" * 70)
    print(f"Database: {settings.database_url.split('@')[-1] if '@' in settings.database_url else settings.database_url}")
    print()
    
    # Check LangGraph checkpoints
    check_langgraph_checkpoints(
        conversation_id=args.conversation_id,
        tenant_id=args.tenant_id,
    )
    
    print()
    
    # Check conversations table
    if args.all or not args.conversation_id:
        check_conversations(
            conversation_id=args.conversation_id,
            tenant_id=args.tenant_id,
            flow_type=args.flow_type,
        )
        print()
    
    # Check messages table
    if args.all or not args.conversation_id:
        check_messages(
            conversation_id=args.conversation_id,
            tenant_id=args.tenant_id,
            flow_type=args.flow_type,
        )
        print()
    
    # Check tool executions
    if args.all:
        check_tool_executions()
        print()
    
    print("=" * 70)
    print("Check Complete")
    print("=" * 70)
    print("\nTo check a specific conversation:")
    print("  python scripts/check_conversations.py --conversation-id YOUR_CONV_ID")
    print("\nTo check all conversations:")
    print("  python scripts/check_conversations.py --all")


if __name__ == "__main__":
    main()

