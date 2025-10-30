"""Repository for conversation storage and retrieval.

Handles PostgreSQL storage for agent conversation history including
messages, tool calls, and conversation metadata.
"""

import logging
from datetime import datetime
from uuid import uuid4

from sqlmodel import Session, select

from app.db.models import Conversation, Message

logger = logging.getLogger(__name__)


class ConversationRepository:
    """Repository for managing conversation data in PostgreSQL.

    Provides methods for creating conversations, saving messages, and
    retrieving conversation history for agent flows.

    Example:
        >>> from sqlmodel import Session
        >>> from app.db.base import get_session
        >>>
        >>> with Session(get_session()) as session:
        ...     repo = ConversationRepository(session)
        ...     conv_id = repo.create_conversation(tenant_id="tenant_123")
        ...     repo.save_message(
        ...         conversation_id=conv_id,
        ...         tenant_id="tenant_123",
        ...         role="user",
        ...         content="Hello!"
        ...     )
        ...     history = repo.get_conversation_history(conv_id)
    """

    def __init__(self, session: Session) -> None:
        """Initialize repository with database session.

        Args:
            session: SQLModel database session
        """
        self.session = session

    def create_conversation(
        self,
        tenant_id: str,
        flow_type: str,
        conversation_id: str | None = None,
        flow_metadata: dict | None = None,
    ) -> str:
        """Create a new conversation.

        Args:
            tenant_id: Tenant identifier for multi-tenancy
            flow_type: Flow type ('weather', 'github', 'slack', etc.)
            conversation_id: Optional conversation ID. If None, generates UUID
            flow_metadata: Optional flow-specific metadata

        Returns:
            Conversation ID (UUID string)

        Example:
            >>> repo = ConversationRepository(session)
            >>> conv_id = repo.create_conversation(
            ...     tenant_id="tenant_123",
            ...     flow_type="weather"
            ... )
            >>> print(conv_id)  # "550e8400-e29b-41d4-a716-446655440000"
        """
        if not conversation_id:
            conversation_id = str(uuid4())

        conversation = Conversation(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            flow_type=flow_type,
            flow_metadata=flow_metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        self.session.add(conversation)
        self.session.commit()

        logger.info(
            f"Created {flow_type} conversation {conversation_id}",
            extra={
                "conversation_id": conversation_id,
                "tenant_id": tenant_id,
                "flow_type": flow_type,
            },
        )

        return conversation_id

    def save_message(
        self,
        conversation_id: str,
        tenant_id: str,
        flow_type: str,
        role: str,
        content: str,
        tool_calls: dict | None = None,
        message_metadata: dict | None = None,
    ) -> Message:
        """Save a message to the conversation.

        Args:
            conversation_id: UUID of the conversation
            tenant_id: Tenant identifier
            flow_type: Flow type ('weather', 'github', 'slack', etc.)
            role: Message role ('user', 'assistant', 'system', 'tool')
            content: Message content
            tool_calls: Optional tool call information (for assistant messages)
            message_metadata: Optional message-specific metadata

        Returns:
            Created Message instance

        Example:
            >>> repo.save_message(
            ...     conversation_id="550e8400-...",
            ...     tenant_id="tenant_123",
            ...     flow_type="weather",
            ...     role="assistant",
            ...     content="The weather is sunny",
            ...     tool_calls={"name": "get_weather", "args": {"location": "London"}}
            ... )
        """
        message = Message(
            conversation_id=conversation_id,
            tenant_id=tenant_id,
            flow_type=flow_type,
            role=role,
            content=content,
            tool_calls=tool_calls,
            message_metadata=message_metadata or {},
            created_at=datetime.utcnow(),
        )

        self.session.add(message)

        # Update conversation updated_at timestamp
        conversation = self.session.exec(
            select(Conversation).where(Conversation.conversation_id == conversation_id)
        ).first()

        if conversation:
            conversation.updated_at = datetime.utcnow()

        self.session.commit()

        logger.debug(
            f"Saved {role} message to {flow_type} conversation {conversation_id}",
            extra={
                "conversation_id": conversation_id,
                "tenant_id": tenant_id,
                "flow_type": flow_type,
                "role": role,
                "has_tool_calls": tool_calls is not None,
            },
        )

        return message

    def get_conversation_history(
        self,
        conversation_id: str,
        limit: int | None = None,
    ) -> list[dict]:
        """Get conversation message history.

        Retrieves all messages in a conversation, ordered by creation time.
        Returns messages in OpenAI chat format for easy LLM consumption.

        Args:
            conversation_id: UUID of the conversation
            limit: Optional limit on number of messages to return (most recent)

        Returns:
            List of message dicts in format: [{"role": "user", "content": "..."}]

        Example:
            >>> history = repo.get_conversation_history("550e8400-...")
            >>> print(history)
            [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny in London"}
            ]
        """
        query = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )

        if limit:
            # Get most recent messages
            query = query.limit(limit)

        messages = self.session.exec(query).all()

        # Convert to OpenAI chat format
        history = []
        for msg in messages:
            message_dict = {
                "role": msg.role,
                "content": msg.content,
            }

            # Include tool_calls if present (for assistant messages)
            if msg.tool_calls:
                message_dict["tool_calls"] = msg.tool_calls

            # Include tool_call_id if present (for tool messages)
            if msg.role == "tool" and msg.message_metadata.get("tool_call_id"):
                message_dict["tool_call_id"] = msg.message_metadata["tool_call_id"]

            history.append(message_dict)

        logger.debug(
            f"Retrieved {len(history)} messages from conversation {conversation_id}",
            extra={
                "conversation_id": conversation_id,
                "message_count": len(history),
            },
        )

        return history

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get conversation by ID.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            Conversation instance or None if not found
        """
        return self.session.exec(
            select(Conversation).where(Conversation.conversation_id == conversation_id)
        ).first()

    def conversation_exists(self, conversation_id: str) -> bool:
        """Check if conversation exists.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            True if conversation exists, False otherwise
        """
        return self.get_conversation(conversation_id) is not None

    def conversation_exists_for_tenant(self, conversation_id: str, tenant_id: str) -> bool:
        """Check if conversation exists and belongs to tenant.

        Args:
            conversation_id: UUID of the conversation
            tenant_id: Tenant identifier

        Returns:
            True if conversation exists and belongs to tenant, False otherwise

        Security:
            This method enforces tenant isolation by validating that the conversation
            belongs to the specified tenant before allowing access.
        """
        conversation = self.session.exec(
            select(Conversation)
            .where(Conversation.conversation_id == conversation_id)
            .where(Conversation.tenant_id == tenant_id)
        ).first()
        return conversation is not None

    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages.

        Args:
            conversation_id: UUID of the conversation

        Returns:
            True if deleted, False if conversation not found
        """
        # Delete messages first
        messages = self.session.exec(
            select(Message).where(Message.conversation_id == conversation_id)
        ).all()

        for msg in messages:
            self.session.delete(msg)

        # Delete conversation
        conversation = self.get_conversation(conversation_id)
        if conversation:
            self.session.delete(conversation)
            self.session.commit()

            logger.info(
                f"Deleted conversation {conversation_id}",
                extra={
                    "conversation_id": conversation_id,
                    "messages_deleted": len(messages),
                },
            )
            return True

        return False
