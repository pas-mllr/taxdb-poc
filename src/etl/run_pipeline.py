async def run_pipeline(
    jurisdiction: str,
    fetch_func: Callable[[date, date], Awaitable[List[DocumentType]]],
    lookback_hours: Optional[int] = None,
    cache_manager: Optional[CacheManager] = None,
    embedding_strategy: Optional[EmbeddingStrategy] = None,
    document_processor: Optional[DocumentProcessor] = None,
    session_factory: Optional[Callable[[], Session]] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Run ETL pipeline for a jurisdiction.
    
    Args:
        jurisdiction: Jurisdiction code (BE, ES, DE)
        fetch_func: Async function that fetches document metadata
        lookback_hours: Hours to look back for documents. Defaults to settings.DOC_LOOKBACK_HOURS.
        cache_manager: Cache manager instance
        embedding_strategy: Embedding strategy instance
        document_processor: Document processor instance
        session_factory: Factory function for database sessions
        metadata: Additional metadata
        
    Returns:
        Dictionary of pipeline statistics
        
    Raises:
        ETLError: If pipeline execution fails
    """
    logger.info(f"Starting ETL pipeline for {jurisdiction}")
    
    # Initialize components
    if cache_manager is None:
        cache_manager = CacheManager()
    
    if embedding_strategy is None:
        embedding_strategy = get_embedding_strategy()
    
    if document_processor is None:
        document_processor = DocumentProcessor(embedding_strategy, cache_manager)
    
    if session_factory is None:
        engine = create_engine(settings.PG_CONNSTR)
        Base.metadata.create_all(engine)
        session_factory = sessionmaker(bind=engine)
    
    # Calculate date range
    lookback_hours = lookback_hours or settings.DOC_LOOKBACK_HOURS
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=lookback_hours)
    
    # Create database session
    with session_factory() as session:
        # Create pipeline context
        context = PipelineContext(
            jurisdiction=jurisdiction,
            start_date=start_date,
            end_date=end_date,
            session=session,
            cache_manager=cache_manager,
            embedding_strategy=embedding_strategy,
            document_processor=document_processor,
            metadata=metadata or {}
        )
        
        # Fetch documents
        try:
            documents = await fetch_func(start_date.date(), end_date.date())
            logger.info(f"Fetched {len(documents)} documents from {jurisdiction}")
        except Exception as e:
            logger.exception(f"Error fetching documents from {jurisdiction}: {e}")
            context.error_count += 1
            return context.get_stats()
        
        if not documents:
            logger.info(f"No documents found for {jurisdiction} in the specified date range")
            return context.get_stats()
        
        # Process documents
        for doc in documents:
            try:
                # Check if document already exists
                if "checksum" in doc:
                    existing = session.execute(
                        select(Document).filter_by(checksum=doc["checksum"])
                    ).scalar_one_or_none()
                    
                    if existing:
                        logger.info(f"Document already exists: {doc.get('id')}")
                        context.skipped_count += 1
                        continue
                
                # Create document object
                document = Document(
                    id=doc["id"],
                    jurisdiction=doc["jurisdiction"],
                    source_system=doc["source_system"],
                    document_type=doc["document_type"],
                    title=doc["title"],
                    summary=doc.get("summary"),
                    issue_date=doc["issue_date"],
                    effective_date=doc.get("effective_date"),
                    language_orig=doc["language_orig"],
                    blob_url=doc["blob_url"],
                    checksum=doc["checksum"],
                    vector=doc.get("vector")
                )
                
                # Save to database
                try:
                    session.add(document)
                    session.commit()
                    context.processed_count += 1
                    logger.info(f"Saved document: {doc['id']}")
                except IntegrityError:
                    session.rollback()
                    logger.warning(f"Document already exists (integrity error): {doc['id']}")
                    context.skipped_count += 1
                except Exception as e:
                    session.rollback()
                    logger.exception(f"Error saving document {doc['id']}: {e}")
                    context.error_count += 1
            except Exception as e:
                logger.exception(f"Error processing document: {e}")
                context.error_count += 1
        
        logger.info(f"Completed ETL pipeline for {jurisdiction}, processed {context.processed_count} documents")
        return context.get_stats()