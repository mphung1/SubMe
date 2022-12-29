import React from "react";
import { Input, InputGroup, InputLeftElement } from "@chakra-ui/react";
import { FaYoutube } from "react-icons/fa";

class YoutubeSearchBar extends React.Component<any, any> {
  handleChange = (event) => {
    this.setState({
      term: event.target.value,
    });
  };

  handleSubmit = (event) => {
    event.preventDefault();
    this.props.handleFormSubmit(this.state.term);
  };

  render() {
    return (
      <>
        <form onSubmit={this.handleSubmit}>
          <div>
            <InputGroup>
              <InputLeftElement pointerEvents="none">
                <FaYoutube color="gray.300" />
              </InputLeftElement>

              <Input
                onChange={this.handleChange}
                name="video-search"
                type="text"
                placeholder="Enter keyword for videos"
              />
            </InputGroup>
          </div>
        </form>
      </>
    );
  }
}

export default YoutubeSearchBar;
