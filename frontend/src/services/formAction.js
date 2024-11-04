
export const handleScroll = (pageRef) => {
    pageRef.current.scrollIntoView({behavior: 'smooth'});
}

export const handleInput = (e, setFormData) => {
    const { id, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [id]: value,
    }));
  };