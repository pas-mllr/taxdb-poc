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